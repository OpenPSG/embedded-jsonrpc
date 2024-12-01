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
//! ### Create an RPC Server
//!
//! ```rust
//! use embedded_jsonrpc::{RpcError, RpcResponse, RpcServer, RpcHandler, JSONRPC_VERSION, DEFAULT_STACK_SIZE};
//! use stackfuture::StackFuture;
//!
//! struct MyHandler;
//!
//! impl RpcHandler for MyHandler {
//!    fn handle<'a>(&self, id: Option<u64>, _request_json: &'a [u8], response_json: &'a mut [u8]) -> StackFuture<'a, Result<usize, RpcError>, DEFAULT_STACK_SIZE> {
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
//! let mut server: RpcServer<'_> = RpcServer::new();
//! server.register_method("echo", &MyHandler);
//! ```
//!
//! ### Serve Requests
//!
//! ```ignore
//! let mut stream: YourAsyncStream = YourAsyncStream::new();
//! server.serve(&mut stream).await.unwrap();
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

#![cfg_attr(not(feature = "std"), no_std)]

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
    pubsub::{PubSubChannel, WaitResult},
};
use embedded_io_async::{Read, Write};
use heapless::{FnvIndexMap, String, Vec};
use serde::{Deserialize, Serialize};
use stackfuture::StackFuture;

#[cfg(feature = "defmt")]
use defmt::*;

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

/// JSON-RPC Response structure
#[derive(Debug, Deserialize, Serialize)]
pub struct RpcResponse<'a, T> {
    pub jsonrpc: &'a str,
    pub id: Option<u64>,
    pub error: Option<RpcError>,
    pub result: Option<T>,
}

/// JSON-RPC Standard Error Codes
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[allow(dead_code)]
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

/// JSON-RPC Error structure
#[derive(Debug, Deserialize, Serialize)]
pub struct RpcError {
    pub code: RpcErrorCode,
    pub message: String<32>,
}

impl RpcError {
    /// Create a new `RpcError` from `RpcErrorCode`
    pub fn from_code(code: RpcErrorCode) -> Self {
        RpcError {
            code,
            message: String::try_from(code.message()).unwrap(),
        }
    }
}

/// Type for errors returned by the RPC server
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub enum RpcServerError {
    /// Buffer overflow error, e.g. message too large
    BufferOverflow,
    /// IO error, e.g. read/write error
    IoError,
    // Parse error, e.g. invalid JSON
    ParseError,
}

/// Default maximum number of clients.
pub const DEFAULT_MAX_CLIENTS: usize = 4;
/// Maximum number of registered RPC methods.
pub const DEFAULT_MAX_HANDLERS: usize = 8;
/// Maximum length of a JSON-RPC message (including headers).
pub const DEFAULT_MAX_MESSAGE_LEN: usize = 512;
/// Default stack size for futures.
/// This is a rough estimate and may need to be adjusted based on the complexity of the handler.
pub const DEFAULT_STACK_SIZE: usize = 256;

/// Trait for RPC handlers
pub trait RpcHandler<const STACK_SIZE: usize = DEFAULT_STACK_SIZE>: Sync {
    fn handle<'a>(
        &self,
        id: Option<u64>,
        request_json: &'a [u8],
        response_json: &'a mut [u8],
    ) -> StackFuture<'a, Result<usize, RpcError>, STACK_SIZE>;
}

/// RPC server
pub struct RpcServer<
    'a,
    const MAX_CLIENTS: usize = DEFAULT_MAX_CLIENTS,
    const MAX_HANDLERS: usize = DEFAULT_MAX_HANDLERS,
    const MAX_MESSAGE_LEN: usize = DEFAULT_MAX_MESSAGE_LEN,
    const STACK_SIZE: usize = DEFAULT_STACK_SIZE,
> {
    handlers: FnvIndexMap<&'a str, &'a dyn RpcHandler<STACK_SIZE>, MAX_HANDLERS>,
    notifications:
        PubSubChannel<CriticalSectionRawMutex, Vec<u8, MAX_MESSAGE_LEN>, 2, MAX_CLIENTS, 1>,
}

impl<'a, const MAX_CLIENTS: usize, const MAX_HANDLERS: usize, const MAX_MESSAGE_LEN: usize> Default
    for RpcServer<'a, MAX_CLIENTS, MAX_HANDLERS, MAX_MESSAGE_LEN>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, const MAX_CLIENTS: usize, const MAX_HANDLERS: usize, const MAX_MESSAGE_LEN: usize>
    RpcServer<'a, MAX_CLIENTS, MAX_HANDLERS, MAX_MESSAGE_LEN>
{
    /// Create a new RPC server
    pub fn new() -> Self {
        Self {
            handlers: FnvIndexMap::new(),
            notifications: PubSubChannel::new(),
        }
    }

    /// Register a new RPC method and its handler
    pub fn register_method(&mut self, name: &'a str, handler: &'a dyn RpcHandler) {
        if !self.handlers.insert(name, handler).is_ok() {
            panic!("Too many handlers registered");
        }
    }

    /// Broadcast a message to all connected clients.
    pub async fn notify(&self, notification_json: &[u8]) -> Result<(), RpcServerError> {
        let mut headers: String<32> = String::new();
        core::fmt::write(
            &mut headers,
            format_args!("Content-Length: {}\r\n\r\n", notification_json.len()),
        )
        .unwrap();
        if headers.len() + notification_json.len() > MAX_MESSAGE_LEN {
            return Err(RpcServerError::BufferOverflow);
        }

        // Construct the framed message
        let mut framed_message: heapless::Vec<u8, MAX_MESSAGE_LEN> = heapless::Vec::new();

        // Add header and payload to the message buffer
        framed_message
            .extend_from_slice(headers.as_bytes())
            .unwrap();
        framed_message.extend_from_slice(notification_json).unwrap();

        // Publish the message to the notification channel
        let notifications = self.notifications.publisher().unwrap();
        notifications.publish(framed_message).await;

        Ok(())
    }

    /// Serve requests using the given stream.
    pub async fn serve<T: Read + Write>(&self, stream: &mut T) -> Result<(), RpcServerError> {
        let mut notifications = self.notifications.subscriber().unwrap();

        let mut request_buffer = [0u8; MAX_MESSAGE_LEN];
        let mut response_json = [0u8; MAX_MESSAGE_LEN];
        let mut read_offset = 0;

        loop {
            let result = select(
                notifications.next_message(),
                stream.read(&mut request_buffer[read_offset..]),
            )
            .await;

            match result {
                Either::First(WaitResult::Message(notification_json)) => {
                    stream
                        .write_all(&notification_json)
                        .await
                        .map_err(|_| RpcServerError::IoError)?;
                    stream.flush().await.map_err(|_| RpcServerError::IoError)?;
                    continue;
                }
                Either::First(WaitResult::Lagged(x)) => {
                    #[cfg(feature = "defmt")]
                    warn!("Dropped {:?} notifications", x);
                }
                Either::Second(Ok(0)) => return Ok(()),
                Either::Second(Ok(n)) => {
                    read_offset += n;

                    // Process complete frames from the buffer.
                    while let Some(headers_len) =
                        Self::parse_headers(&request_buffer[..read_offset])
                    {
                        let content_len: usize =
                            Self::parse_content_length(&mut request_buffer[..headers_len])?;
                        let total_message_len = headers_len + content_len;

                        if read_offset < total_message_len {
                            // Not enough data for a complete message; wait for more.
                            break;
                        }

                        // Process the complete JSON-RPC message.
                        let request_json = &request_buffer[headers_len..headers_len + content_len];
                        let response_json_len =
                            self.handle_request(request_json, &mut response_json).await;

                        // Construct the response
                        let mut headers: String<32> = String::new();
                        core::fmt::write(
                            &mut headers,
                            format_args!("Content-Length: {}\r\n\r\n", response_json_len),
                        )
                        .unwrap();

                        if headers.len() + response_json_len > MAX_MESSAGE_LEN {
                            return Err(RpcServerError::BufferOverflow);
                        }

                        // Write the headers and response to the stream
                        stream
                            .write_all(headers.as_bytes())
                            .await
                            .map_err(|_| RpcServerError::IoError)?;
                        stream
                            .write_all(&response_json[..response_json_len])
                            .await
                            .map_err(|_| RpcServerError::IoError)?;
                        stream.flush().await.map_err(|_| RpcServerError::IoError)?;

                        // Remove the processed message from the buffer.
                        let remaining = read_offset - total_message_len;
                        request_buffer.copy_within(total_message_len..read_offset, 0);
                        read_offset = remaining;
                    }
                }
                Either::Second(Err(_)) => return Err(RpcServerError::IoError),
            }
        }
    }

    /// Handle a single JSON-RPC request
    async fn handle_request(&self, request_json: &'a [u8], response_json: &'a mut [u8]) -> usize {
        let request: RpcRequest<'_, ()> = match serde_json_core::from_slice(request_json) {
            Ok((request, _remainder)) => request,
            Err(_) => {
                if let Ok(json_str) = core::str::from_utf8(request_json) {
                    #[cfg(feature = "defmt")]
                    warn!("Invalid JSON-RPC request: {}", json_str)
                } else {
                    #[cfg(feature = "defmt")]
                    warn!("Invalid JSON-RPC request: [non-UTF8 data]")
                }

                let response: RpcResponse<'_, ()> = RpcResponse {
                    jsonrpc: JSONRPC_VERSION,
                    error: Some(RpcError::from_code(RpcErrorCode::ParseError)),
                    id: None,
                    result: None,
                };

                return serde_json_core::to_slice(&response, &mut response_json[..]).unwrap();
            }
        };

        let id = request.id;

        if request.jsonrpc != JSONRPC_VERSION {
            let response: RpcResponse<'_, ()> = RpcResponse {
                jsonrpc: JSONRPC_VERSION,
                error: Some(RpcError::from_code(RpcErrorCode::InvalidRequest)),
                result: None,
                id,
            };

            return serde_json_core::to_slice(&response, &mut response_json[..]).unwrap();
        }

        return match self.handlers.get(request.method) {
            Some(handler) => match handler.handle(id, request_json, response_json).await {
                Ok(response_len) => response_len,
                Err(e) => {
                    let response: RpcResponse<'_, ()> = RpcResponse {
                        jsonrpc: JSONRPC_VERSION,
                        error: Some(e),
                        result: None,
                        id,
                    };

                    serde_json_core::to_slice(&response, &mut response_json[..]).unwrap()
                }
            },
            None => {
                let response: RpcResponse<'_, ()> = RpcResponse {
                    jsonrpc: JSONRPC_VERSION,
                    error: Some(RpcError::from_code(RpcErrorCode::MethodNotFound)),
                    result: None,
                    id,
                };

                serde_json_core::to_slice(&response, &mut response_json[..]).unwrap()
            }
        };
    }

    /// Parse the headers of the message, returning the index of the end of the headers.
    fn parse_headers(buffer: &[u8]) -> Option<usize> {
        return buffer
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .map(|i| i + 4);
    }

    /// Extract the Content-Length value from headers.
    fn parse_content_length(buffer: &mut [u8]) -> Result<usize, RpcServerError> {
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

    mod memory_pipe;

    #[tokio::test]
    async fn test_request_response() {
        let mut server: RpcServer<'_> = RpcServer::new();
        server.register_method("echo", &EchoHandler);

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
        let server: Arc<RpcServer<'_>> = Arc::new(RpcServer::new());

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
            _request_json: &'a [u8],
            response_json: &'a mut [u8],
        ) -> StackFuture<'a, Result<usize, RpcError>, DEFAULT_STACK_SIZE> {
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
