//! # JSON-RPC for Embedded Systems
//!
//! This crate provides a JSON-RPC server implementation for embedded systems.
//!
//! ## Features
//!
//! - **`#![no_std]` Support**: Fully compatible with environments lacking a standard library.
//! - **Predictable Memory Usage**: Zero dynamic allocation with statically sized buffers.
//! - **Async**: Non-blocking I/O with `embedded-io-async`.
//! - **Client Compatibility**: Uses LSP style framing for JSON-RPC messages.
//! - **Error Handling**: Adheres to JSON-RPC standards with robust error reporting.
//!
//! ## Example Usage
//!
//! ```rust
//! use embedded_jsonrpc::{RpcError, RpcResponse, RpcServer, RpcHandler, JSONRPC_VERSION, DEFAULT_HANDLER_STACK_SIZE};
//! use embedded_jsonrpc::stackfuture::StackFuture;
//! use embedded_io_async::{Read, Write, ErrorType};
//!
//! struct MyStream;
//!
//! impl ErrorType for MyStream {
//!   type Error = embedded_io_async::ErrorKind;
//! }
//!
//! impl Read for MyStream {
//!   async fn read(&mut self, buf: &mut [u8]) -> Result<usize, Self::Error> {
//!     Ok(0)
//!   }
//! }
//!
//! impl Write for MyStream {
//!  async fn write(&mut self, buf: &[u8]) -> Result<usize, Self::Error> {
//!   Ok(0)
//!  }
//! }
//!
//! struct MyHandler;
//!
//! impl RpcHandler for MyHandler {
//!    fn handle<'a>(&self, id: Option<u64>, _method: &'a str, _request_json: &'a [u8], response_json: &'a mut [u8]) -> StackFuture<'a, Result<usize, RpcError>, DEFAULT_HANDLER_STACK_SIZE> {
//!       StackFuture::from(async move {
//!          let response: RpcResponse<'static, ()> = RpcResponse {
//!            jsonrpc: JSONRPC_VERSION,
//!            error: None,
//!            result: None,
//!            id,
//!          };
//!         Ok(serde_json_core::to_slice(&response, response_json).unwrap())
//!      })
//!   }
//! }
//!
//! async fn serve_requests() {
//!   let mut server: RpcServer<'_, _> = RpcServer::new();
//!   server.register_handler("echo", &MyHandler);
//!
//!   loop {
//!     let mut stream: MyStream = MyStream;
//!     server.serve(&mut stream).await.unwrap();
//!   }
//! }
//! ```
//!
//! ## License
//!
//! This crate is licensed under the Mozilla Public License 2.0 (MPL-2.0).
//! See the LICENSE file for more details.
//!
//! ## References
//!
//! - [JSON-RPC 2.0 Specification](https://www.jsonrpc.org/specification)
//! - [Protocol Buffers Varint Encoding](https://protobuf.dev/programming-guides/encoding/#varints)
//! - [Embedded IO Async](https://docs.rs/embedded-io-async)
//!

#![cfg_attr(not(test), no_std)]

use crate::glob_match::glob_match;
use crate::stackfuture::StackFuture;
use core::clone::Clone;
use core::cmp::{Eq, PartialEq};
use core::default::Default;
use core::fmt::Debug;
use core::format_args;
use core::iter::Iterator;
use core::marker::Copy;
use core::option::Option::{self, *};
use core::prelude::v1::derive;
use core::result::Result::{self, *};
use embassy_futures::select::{select, Either};
use embassy_sync::{
    blocking_mutex::raw::CriticalSectionRawMutex,
    mutex::Mutex,
    pubsub::{PubSubChannel, WaitResult},
};
use embedded_io_async::{Read, Write};
use heapless::{FnvIndexMap, String, Vec};
use serde::{Deserialize, Serialize};

#[cfg(feature = "defmt")]
use defmt::{debug, error, warn};

#[cfg(feature = "embassy-time")]
use embassy_time::{with_timeout, Duration};

mod glob_match;
pub mod stackfuture;

/// Default maximum number of clients.
pub const DEFAULT_MAX_CLIENTS: usize = 4;
/// Maximum number of registered RPC methods.
pub const DEFAULT_MAX_HANDLERS: usize = 8;
/// Maximum length of a JSON-RPC message (including headers).
/// Default to the largest message that'll fit in a single Ethernet frame.
pub const DEFAULT_MAX_MESSAGE_LEN: usize = 1460;
/// Default stack size for futures.
/// This is a rough estimate and will need to be adjusted based on the complexity of your handlers.
pub const DEFAULT_HANDLER_STACK_SIZE: usize = 256;
/// Default notification queue size.
/// This is the maximum number of notifications that can be queued for sending.
pub const DEFAULT_NOTIFICATION_QUEUE_SIZE: usize = 1;
/// Default write timeout in milliseconds.
/// This is the maximum time allowed for writing to the stream.
pub const DEFAULT_WRITE_TIMEOUT_MS: u64 = 5000;
/// Default handler timeout in milliseconds.
/// This is the maximum time allowed for a handler to process a request.
pub const DEFAULT_HANDLER_TIMEOUT_MS: u64 = 5000;

/// JSON-RPC Version
/// Currently only supports version 2.0
/// https://www.jsonrpc.org/specification
pub const JSONRPC_VERSION: &str = "2.0";

/// JSON-RPC Request structure
#[derive(Debug, Deserialize, Serialize)]
pub struct RpcRequest<'a, T> {
    pub jsonrpc: &'a str,
    pub id: Option<u64>,
    pub method: &'a str,
    pub params: Option<T>,
}

#[derive(Debug, Deserialize)]
struct RpcRequestMetadata<'a> {
    pub jsonrpc: &'a str,
    pub id: Option<u64>,
    pub method: &'a str,
}

/// JSON-RPC Response structure
#[derive(Debug, Deserialize, Serialize)]
pub struct RpcResponse<'a, T> {
    pub jsonrpc: &'a str,
    pub id: Option<u64>,
    pub error: Option<RpcError>,
    pub result: Option<T>,
}

/// JSON-RPC Standard Error Codes
#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub enum RpcErrorCode {
    ParseError = -32700,
    InvalidRequest = -32600,
    MethodNotFound = -32601,
    InvalidParams = -32602,
    InternalError = -32603,
}

impl RpcErrorCode {
    /// Get the standard message for the error code.
    pub fn message(self) -> &'static str {
        match self {
            RpcErrorCode::ParseError => "Invalid JSON.",
            RpcErrorCode::InvalidRequest => "Invalid request.",
            RpcErrorCode::MethodNotFound => "Method not found.",
            RpcErrorCode::InvalidParams => "Invalid parameters.",
            RpcErrorCode::InternalError => "Internal error.",
        }
    }
}

impl Serialize for RpcErrorCode {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::ser::Serializer,
    {
        (*self as i32).serialize(serializer)
    }
}

impl<'a> Deserialize<'a> for RpcErrorCode {
    fn deserialize<D>(deserializer: D) -> Result<RpcErrorCode, D::Error>
    where
        D: serde::de::Deserializer<'a>,
    {
        let code = i32::deserialize(deserializer)?;
        match code {
            -32700 => Ok(RpcErrorCode::ParseError),
            -32600 => Ok(RpcErrorCode::InvalidRequest),
            -32601 => Ok(RpcErrorCode::MethodNotFound),
            -32602 => Ok(RpcErrorCode::InvalidParams),
            -32603 => Ok(RpcErrorCode::InternalError),
            _ => Err(serde::de::Error::custom("Invalid error code")),
        }
    }
}

/// JSON-RPC Error structure
#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub struct RpcError {
    pub code: RpcErrorCode,
    pub message: String<32>,
}

impl From<RpcErrorCode> for RpcError {
    fn from(code: RpcErrorCode) -> Self {
        RpcError {
            code,
            message: String::try_from(code.message()).unwrap(),
        }
    }
}

/// Type for errors returned by the RPC server
#[derive(PartialEq, Eq, Clone, Debug)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub enum RpcServerError<E> {
    /// Buffer overflow error, e.g., message too large.
    BufferOverflow,
    /// IO error, e.g., read/write error.
    IoError(E),
    /// Parse error, e.g., invalid JSON.
    ParseError,
    /// Too many registered handlers
    /// The maximum number of handlers is defined by `MAX_HANDLERS`.
    TooManyHandlers,
    /// Timeout error
    /// The operation took longer than the specified timeout.
    TimeoutError,
}

/// Trait for RPC handlers
/// TODO: when async closures are stabilized, we should offer a way to register
/// closures directly.
pub trait RpcHandler<const STACK_SIZE: usize = DEFAULT_HANDLER_STACK_SIZE>: Sync {
    fn handle<'a>(
        &'a self,
        id: Option<u64>,
        method: &'a str,
        request_json: &'a [u8],
        response_json: &'a mut [u8],
    ) -> StackFuture<'a, Result<usize, RpcError>, STACK_SIZE>;
}

/// RPC server
pub struct RpcServer<
    'a,
    StreamError,
    const MAX_CLIENTS: usize = DEFAULT_MAX_CLIENTS,
    const MAX_HANDLERS: usize = DEFAULT_MAX_HANDLERS,
    const MAX_MESSAGE_LEN: usize = DEFAULT_MAX_MESSAGE_LEN,
    const HANDLER_STACK_SIZE: usize = DEFAULT_HANDLER_STACK_SIZE,
    const NOTIFICATION_QUEUE_SIZE: usize = DEFAULT_NOTIFICATION_QUEUE_SIZE,
> {
    handlers: FnvIndexMap<&'a str, &'a dyn RpcHandler<HANDLER_STACK_SIZE>, MAX_HANDLERS>,
    notifications: PubSubChannel<
        CriticalSectionRawMutex,
        Vec<u8, MAX_MESSAGE_LEN>,
        NOTIFICATION_QUEUE_SIZE,
        MAX_CLIENTS,
        1,
    >,
    notification_publisher_mutex: Mutex<CriticalSectionRawMutex, ()>,
    _phantom: core::marker::PhantomData<StreamError>,
}

impl<
        StreamError,
        const MAX_CLIENTS: usize,
        const MAX_HANDLERS: usize,
        const MAX_MESSAGE_LEN: usize,
        const HANDLER_STACK_SIZE: usize,
        const NOTIFICATION_QUEUE_SIZE: usize,
    > Default
    for RpcServer<
        '_,
        StreamError,
        MAX_CLIENTS,
        MAX_HANDLERS,
        MAX_MESSAGE_LEN,
        HANDLER_STACK_SIZE,
        NOTIFICATION_QUEUE_SIZE,
    >
{
    fn default() -> Self {
        Self::new()
    }
}

impl<
        'a,
        StreamError,
        const MAX_CLIENTS: usize,
        const MAX_HANDLERS: usize,
        const MAX_MESSAGE_LEN: usize,
        const HANDLER_STACK_SIZE: usize,
        const NOTIFICATION_QUEUE_SIZE: usize,
    >
    RpcServer<
        'a,
        StreamError,
        MAX_CLIENTS,
        MAX_HANDLERS,
        MAX_MESSAGE_LEN,
        HANDLER_STACK_SIZE,
        NOTIFICATION_QUEUE_SIZE,
    >
{
    /// Create a new RPC server
    pub fn new() -> Self {
        #[cfg(feature = "defmt")]
        debug!("Initializing new RPC server");

        Self {
            handlers: FnvIndexMap::new(),
            notifications: PubSubChannel::new(),
            notification_publisher_mutex: Mutex::new(()),
            _phantom: core::marker::PhantomData,
        }
    }

    /// Register a new RPC method and its handler
    pub fn register_handler(
        &mut self,
        method_name_glob: &'a str,
        handler: &'a dyn RpcHandler<HANDLER_STACK_SIZE>,
    ) -> Result<(), RpcServerError<StreamError>> {
        #[cfg(feature = "defmt")]
        debug!("Registering method: {}", method_name_glob);

        if self.handlers.insert(method_name_glob, handler).is_err() {
            #[cfg(feature = "defmt")]
            warn!(
                "Failed to register method (too many handlers): {}",
                method_name_glob
            );
            return Err(RpcServerError::TooManyHandlers);
        }

        Ok(())
    }

    /// Broadcast a message to all connected clients.
    pub async fn notify(
        &self,
        notification_json: &[u8],
    ) -> Result<(), RpcServerError<StreamError>> {
        #[cfg(feature = "defmt")]
        debug!("Broadcasting notification");

        let mut headers: String<32> = String::new();
        core::fmt::write(
            &mut headers,
            format_args!("Content-Length: {}\r\n\r\n", notification_json.len()),
        )
        .unwrap();

        if headers.len() + notification_json.len() > MAX_MESSAGE_LEN {
            #[cfg(feature = "defmt")]
            error!("Broadcast message too large");
            return Err(RpcServerError::BufferOverflow);
        }

        let mut framed_message: heapless::Vec<u8, MAX_MESSAGE_LEN> = heapless::Vec::new();
        framed_message
            .extend_from_slice(headers.as_bytes())
            .unwrap();
        framed_message.extend_from_slice(notification_json).unwrap();

        {
            let _lock = self.notification_publisher_mutex.lock().await;
            let notifications = self.notifications.publisher().unwrap();
            notifications.publish(framed_message).await;
        }

        Ok(())
    }

    /// Serve requests using the given stream.
    pub async fn serve<Stream: Read<Error = StreamError> + Write<Error = StreamError>>(
        &self,
        stream: &mut Stream,
    ) -> Result<(), RpcServerError<StreamError>> {
        #[cfg(feature = "defmt")]
        debug!("Starting RPC server");

        let mut notifications = self.notifications.subscriber().unwrap();
        let mut request_buffer = [0u8; MAX_MESSAGE_LEN];
        let mut response_json = [0u8; MAX_MESSAGE_LEN];
        let mut read_offset = 0;

        loop {
            #[cfg(feature = "defmt")]
            debug!("Waiting for data from client");

            let result = select(
                notifications.next_message(),
                stream.read(&mut request_buffer[read_offset..]),
            )
            .await;

            match result {
                Either::First(WaitResult::Message(notification_json)) => {
                    #[cfg(feature = "defmt")]
                    debug!("Writing notification");

                    #[cfg(feature = "embassy-time")]
                    {
                        with_timeout(
                            Duration::from_millis(DEFAULT_WRITE_TIMEOUT_MS),
                            stream.write_all(&notification_json),
                        )
                        .await
                        .map_err(|_| RpcServerError::TimeoutError)?
                        .map_err(RpcServerError::IoError)?;

                        with_timeout(
                            Duration::from_millis(DEFAULT_WRITE_TIMEOUT_MS),
                            stream.flush(),
                        )
                        .await
                        .map_err(|_| RpcServerError::TimeoutError)?
                        .map_err(RpcServerError::IoError)?;
                    }

                    #[cfg(not(feature = "embassy-time"))]
                    {
                        stream
                            .write_all(&notification_json)
                            .await
                            .map_err(RpcServerError::IoError)?;

                        stream.flush().await.map_err(RpcServerError::IoError)?;
                    }

                    #[cfg(feature = "defmt")]
                    debug!("Notification sent to client");

                    continue;
                }
                Either::First(WaitResult::Lagged(x)) => {
                    #[cfg(feature = "defmt")]
                    warn!("Dropped {} notifications due to lag", x);
                }
                Either::Second(Ok(0)) => {
                    #[cfg(feature = "defmt")]
                    debug!("Client disconnected");
                    return Ok(());
                }
                Either::Second(Ok(n)) => {
                    #[cfg(feature = "defmt")]
                    debug!("Received {} bytes from client", n);

                    read_offset += n;
                    while let Some(headers_len) =
                        Self::parse_headers(&request_buffer[..read_offset])
                    {
                        let content_len =
                            Self::parse_content_length(&mut request_buffer[..headers_len])?;
                        let total_message_len = headers_len + content_len;

                        if read_offset < total_message_len {
                            #[cfg(feature = "defmt")]
                            debug!("Incomplete message, waiting for more data");
                            break;
                        }

                        #[cfg(feature = "defmt")]
                        debug!("Received complete message, handling request");

                        let request_json = &request_buffer[headers_len..headers_len + content_len];
                        let response_json_len = self
                            .handle_request(request_json, &mut response_json)
                            .await?;

                        #[cfg(feature = "defmt")]
                        debug!("Sending response to client");

                        let mut headers: String<32> = String::new();
                        core::fmt::write(
                            &mut headers,
                            format_args!("Content-Length: {}\r\n\r\n", response_json_len),
                        )
                        .unwrap();

                        if headers.len() + response_json_len > MAX_MESSAGE_LEN {
                            #[cfg(feature = "defmt")]
                            error!("Response message too large");
                            return Err(RpcServerError::BufferOverflow);
                        }

                        #[cfg(feature = "defmt")]
                        debug!("Writing response");

                        #[cfg(feature = "embassy-time")]
                        {
                            with_timeout(
                                Duration::from_millis(DEFAULT_WRITE_TIMEOUT_MS),
                                stream.write_all(headers.as_bytes()),
                            )
                            .await
                            .map_err(|_| RpcServerError::TimeoutError)?
                            .map_err(RpcServerError::IoError)?;

                            with_timeout(
                                Duration::from_millis(DEFAULT_WRITE_TIMEOUT_MS),
                                stream.write_all(&response_json[..response_json_len]),
                            )
                            .await
                            .map_err(|_| RpcServerError::TimeoutError)?
                            .map_err(RpcServerError::IoError)?;

                            with_timeout(
                                Duration::from_millis(DEFAULT_WRITE_TIMEOUT_MS),
                                stream.flush(),
                            )
                            .await
                            .map_err(|_| RpcServerError::TimeoutError)?
                            .map_err(RpcServerError::IoError)?;
                        }

                        #[cfg(not(feature = "embassy-time"))]
                        {
                            stream
                                .write_all(headers.as_bytes())
                                .await
                                .map_err(RpcServerError::IoError)?;

                            stream
                                .write_all(&response_json[..response_json_len])
                                .await
                                .map_err(RpcServerError::IoError)?;

                            stream.flush().await.map_err(RpcServerError::IoError)?;
                        }

                        #[cfg(feature = "defmt")]
                        debug!("Response sent to client");
                        let remaining = read_offset - total_message_len;
                        request_buffer.copy_within(total_message_len..read_offset, 0);
                        read_offset = remaining;
                    }
                }
                Either::Second(Err(e)) => {
                    #[cfg(feature = "defmt")]
                    error!("IO error during stream read");
                    return Err(RpcServerError::IoError(e));
                }
            }
        }
    }

    /// Handle a single JSON-RPC request
    async fn handle_request(
        &self,
        request_json: &'a [u8],
        response_json: &'a mut [u8],
    ) -> Result<usize, RpcServerError<StreamError>> {
        #[cfg(feature = "defmt")]
        debug!("Handling request");

        let request: RpcRequestMetadata = match serde_json_core::from_slice(request_json) {
            Ok((request, _remainder)) => request,
            Err(_) => {
                #[cfg(feature = "defmt")]
                warn!("Failed to parse request JSON");

                let response: RpcResponse<'_, ()> = RpcResponse {
                    jsonrpc: JSONRPC_VERSION,
                    error: Some(RpcErrorCode::ParseError.into()),
                    id: None,
                    result: None,
                };

                return Ok(serde_json_core::to_slice(&response, &mut response_json[..]).unwrap());
            }
        };

        let id = request.id;

        if request.jsonrpc != JSONRPC_VERSION {
            #[cfg(feature = "defmt")]
            warn!("Unsupported JSON-RPC version");

            let response: RpcResponse<'_, ()> = RpcResponse {
                jsonrpc: JSONRPC_VERSION,
                error: Some(RpcErrorCode::InvalidRequest.into()),
                result: None,
                id,
            };

            return Ok(serde_json_core::to_slice(&response, &mut response_json[..]).unwrap());
        }

        #[cfg(feature = "defmt")]
        debug!("Dispatching method: {}", request.method);

        let mut handler: Option<&dyn RpcHandler<HANDLER_STACK_SIZE>> = None;
        for (method_name_glob, h) in self.handlers.iter() {
            if glob_match(method_name_glob, request.method) {
                #[cfg(feature = "defmt")]
                debug!("Matched method: {}", method_name_glob);

                handler = Some(*h);
            }
        }

        if handler.is_none() {
            #[cfg(feature = "defmt")]
            warn!("Method not found: {}", request.method);

            let response: RpcResponse<'_, ()> = RpcResponse {
                jsonrpc: JSONRPC_VERSION,
                error: Some(RpcErrorCode::MethodNotFound.into()),
                result: None,
                id,
            };

            return Ok(serde_json_core::to_slice(&response, &mut response_json[..]).unwrap());
        }

        #[cfg(feature = "embassy-time")]
        let result = with_timeout(
            Duration::from_millis(DEFAULT_HANDLER_TIMEOUT_MS),
            handler.handle(id, request.method, request_json, response_json),
        )
        .await
        .map_err(|_| RpcServerError::TimeoutError)?;

        #[cfg(not(feature = "embassy-time"))]
        let result = handler
            .unwrap()
            .handle(id, request.method, request_json, response_json)
            .await;

        match result {
            Ok(response_len) => Ok(response_len),
            Err(e) => {
                #[cfg(feature = "defmt")]
                error!("Handler returned error: {:?}", e);

                let response: RpcResponse<'_, ()> = RpcResponse {
                    jsonrpc: JSONRPC_VERSION,
                    error: Some(e),
                    result: None,
                    id,
                };

                Ok(serde_json_core::to_slice(&response, &mut response_json[..]).unwrap())
            }
        }
    }

    /// Parse the headers of the message, returning the index of the end of the headers.
    fn parse_headers(buffer: &[u8]) -> Option<usize> {
        buffer
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .map(|i| i + 4)
    }

    /// Extract the Content-Length value from headers.
    fn parse_content_length(buffer: &mut [u8]) -> Result<usize, RpcServerError<StreamError>> {
        let headers = core::str::from_utf8_mut(buffer).map_err(|_| RpcServerError::ParseError)?;
        headers.make_ascii_lowercase();
        for line in headers.lines() {
            if let Some(value) = line.strip_prefix("content-length:") {
                return value.trim().parse().map_err(|_| RpcServerError::ParseError);
            }
        }
        Err(RpcServerError::ParseError)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use memory_pipe::MemoryPipe;
    use std::sync::Arc;

    #[cfg(feature = "defmt")]
    use defmt_logger_tcp as _;

    mod memory_pipe;

    #[tokio::test]
    async fn test_request_response() {
        let mut server: RpcServer<'_, _> = RpcServer::new();
        server.register_handler("echo", &EchoHandler).unwrap();

        let (mut stream1, mut stream2) = MemoryPipe::new();

        tokio::spawn(async move {
            server.serve(&mut stream2).await.unwrap();
        });

        let request: RpcRequest<'_, ()> = RpcRequest {
            jsonrpc: JSONRPC_VERSION,
            id: Some(1),
            method: "echo",
            params: None,
        };

        let mut request_json = [0u8; 256];
        let request_len = serde_json_core::to_slice(&request, &mut request_json).unwrap();

        // Write the request to the stream
        let request_message = format!(
            "Content-Length: {}\r\n\r\n{}",
            request_len,
            core::str::from_utf8(&request_json[..request_len]).unwrap()
        );
        stream1.write_all(request_message.as_bytes()).await.unwrap();

        // Read the response
        let mut response_buffer = [0u8; DEFAULT_MAX_MESSAGE_LEN];
        let response_len = stream1.read(&mut response_buffer).await.unwrap();

        let response = core::str::from_utf8(&response_buffer[..response_len]).unwrap();

        assert_eq!(
            response,
            "Content-Length: 51\r\n\r\n{\"jsonrpc\":\"2.0\",\"id\":1,\"error\":null,\"result\":null}"
        );
    }

    #[tokio::test]
    async fn test_notify() {
        let server: Arc<RpcServer<'_, _>> = Arc::new(RpcServer::new());

        let server_clone = Arc::clone(&server); // Clone for use in the spawned task
        let (mut stream1, mut stream2) = MemoryPipe::new();

        // Spawn the server task to handle notifications
        tokio::spawn(async move {
            server_clone.serve(&mut stream2).await.unwrap();
        });

        // Sleep to allow the server task to start and subscribe to notifications
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Notification to send
        let notification: RpcRequest<'_, ()> = RpcRequest {
            jsonrpc: JSONRPC_VERSION,
            method: "notify",
            id: None,
            params: None,
        };

        let mut notification_json = [0u8; DEFAULT_MAX_MESSAGE_LEN];
        let notification_len =
            serde_json_core::to_slice(&notification, &mut notification_json).unwrap();

        // Notify all clients
        server
            .notify(&notification_json[..notification_len])
            .await
            .unwrap();

        // Read the notification from the stream
        let mut notification_json = [0u8; DEFAULT_MAX_MESSAGE_LEN];
        let notification_len = stream1.read(&mut notification_json).await.unwrap();

        let notification_json =
            core::str::from_utf8(&notification_json[..notification_len]).unwrap();

        assert_eq!(
            notification_json,
            "Content-Length: 59\r\n\r\n{\"jsonrpc\":\"2.0\",\"id\":null,\"method\":\"notify\",\"params\":null}",
        );
    }

    struct EchoHandler;

    impl RpcHandler for EchoHandler {
        fn handle<'a>(
            &self,
            id: Option<u64>,
            _method: &'a str,
            _request_json: &'a [u8],
            response_json: &'a mut [u8],
        ) -> StackFuture<'a, Result<usize, RpcError>, DEFAULT_HANDLER_STACK_SIZE> {
            StackFuture::from(async move {
                let response: RpcResponse<'static, ()> = RpcResponse {
                    jsonrpc: JSONRPC_VERSION,
                    error: None,
                    result: None,
                    id,
                };

                Ok(serde_json_core::to_slice(&response, response_json).unwrap())
            })
        }
    }
}
