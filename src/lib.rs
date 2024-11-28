//! # JSON-RPC for Embedded Systems
//!
//! This crate provides a simple JSON-RPC server implementation for embedded systems.
//!
//! ## Features
//!
//! - **no_std Support**: Compatible with `#![no_std]` environments.
//! - **Zero Alloc**: Uses statically allocated buffers and `heapless` containers for predictable memory usage.
//! - **Async**: Non-blocking IO with `embedded-io-async`.
//! - **Varint Framing**: Efficient message framing and length-prefix encoding using protobuf style varints.
//! - **Error Handling**: Includes comprehensive error codes and descriptions based on the JSON-RPC specification.
//!
//! ## Core Components
//!
//! ### JSON-RPC Data Structures
//!
//! - **`RpcRequest`**: Represents a JSON-RPC request, including the method and optional parameters.
//! - **`RpcResponse`**: Represents a JSON-RPC response, including the result or error information.
//! - **`RpcError` and `RpcErrorCode`**: Encapsulate errors with standard codes and descriptions.
//!
//! ### RPC Server
//!
//! - **`RpcServer`**: Core server to manage method registration and handle incoming requests.
//! - **Handlers**: Define functions for specific RPC methods using a type-safe `RpcHandler`.
//! - **`serve`**: Processes incoming requests and sends responses over an `embedded-io-async` stream.
//!
//! ## Example Usage
//!
//! ### Create an RPC Server
//!
//! ```rust
//! use embedded_jsonrpc::{RpcServer, RpcResponse, RpcError, RpcErrorCode, JSONRPC_VERSION};
//!
//! const MAX_CLIENTS: usize = 1;
//! const MAX_HANDLERS: usize = 2;
//! const MAX_MESSAGE_LEN: usize = 256;
//!
//! let mut server: RpcServer<'_, MAX_CLIENTS, MAX_HANDLERS, MAX_MESSAGE_LEN> = RpcServer::new();
//! server.register_method("echo", |id, _request_json, response_json| {
//!    let response = RpcResponse {
//!        jsonrpc: JSONRPC_VERSION,
//!        id,
//!        error: None,
//!    };
//!    serde_json_core::to_slice(&response, response_json).unwrap()
//! });
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

use crate::varint::{decode as varint_decode, encode as varint_encode};
use embassy_futures::select::{select, Either};
use embassy_sync::{
    blocking_mutex::raw::CriticalSectionRawMutex,
    pubsub::{PubSubChannel, WaitResult},
};
use embedded_io_async::{Read, Write};
use heapless::{FnvIndexMap, Vec};
use serde::{Deserialize, Serialize};

#[cfg(feature = "defmt")]
use defmt::*;

pub mod varint;

/// JSON-RPC Version
/// Currently only supports version 2.0
/// https://www.jsonrpc.org/specification
pub const JSONRPC_VERSION: &str = "2.0";

/// Maximum number of queued, unsent, notifications.
const MAX_QUEUED_NOTIFICATIONS: usize = 4;

/// JSON-RPC Request structure
#[derive(Deserialize, Serialize)]
pub struct RpcRequest<'a> {
    pub jsonrpc: &'a str,
    pub id: Option<u64>,
    pub method: &'a str,
}

/// JSON-RPC Response structure
#[derive(Deserialize, Serialize)]
pub struct RpcResponse<'a> {
    pub jsonrpc: &'a str,
    pub id: Option<u64>,
    pub error: Option<RpcError<'a>>,
}

/// JSON-RPC Standard Error Codes
#[derive(Clone, Copy, Deserialize, Serialize)]
#[allow(dead_code)]
pub enum RpcErrorCode {
    ParseError = -32700,
    InvalidRequest = -32600,
    MethodNotFound = -32601,
    InvalidParams = -32602,
    InternalError = -32603,
}

impl RpcErrorCode {
    /// Get the standard message for the error code
    pub fn message(self) -> &'static str {
        match self {
            RpcErrorCode::ParseError => "Parse error: Invalid JSON received by the server.",
            RpcErrorCode::InvalidRequest => {
                "Invalid Request: The JSON sent is not a valid Request object."
            }
            RpcErrorCode::MethodNotFound => {
                "Method not found: The method does not exist or is not available."
            }
            RpcErrorCode::InvalidParams => "Invalid params: Invalid method parameter(s).",
            RpcErrorCode::InternalError => "Internal error: Internal JSON-RPC error.",
        }
    }
}

/// JSON-RPC Error structure
#[derive(Deserialize, Serialize)]
pub struct RpcError<'a> {
    pub code: RpcErrorCode,
    pub message: &'a str,
}

impl<'a> RpcError<'a> {
    /// Create a new `RpcError` from `RpcErrorCode`
    pub fn from_code(code: RpcErrorCode) -> Self {
        RpcError {
            code,
            message: code.message(),
        }
    }
}

/// Type for RPC handler functions
pub type RpcHandler = fn(id: Option<u64>, request_json: &[u8], response_json: &mut [u8]) -> usize;

/// RPC server
pub struct RpcServer<
    'a,
    const MAX_CLIENTS: usize,
    const MAX_HANDLERS: usize,
    const MAX_MESSAGE_LEN: usize,
> {
    handlers: FnvIndexMap<&'a str, RpcHandler, MAX_HANDLERS>,
    notifications: PubSubChannel<
        CriticalSectionRawMutex,
        Vec<u8, MAX_MESSAGE_LEN>,
        MAX_QUEUED_NOTIFICATIONS,
        MAX_CLIENTS,
        1,
    >,
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
    pub fn register_method(&mut self, name: &'a str, handler: RpcHandler) {
        self.handlers.insert(name, handler).unwrap();
    }

    /// Handle a single JSON-RPC request
    pub fn handle_request(&self, request_json: &[u8], response_json: &mut [u8]) -> usize {
        let request: RpcRequest = match serde_json_core::from_slice::<RpcRequest<'_>>(request_json)
        {
            Ok((request, _remainder)) => request,
            Err(_) => {
                if let Ok(json_str) = core::str::from_utf8(request_json) {
                    #[cfg(feature = "defmt")]
                    warn!("Invalid JSON-RPC request: {}", json_str)
                } else {
                    #[cfg(feature = "defmt")]
                    warn!("Invalid JSON-RPC request: [non-UTF8 data]")
                }

                let response = RpcResponse {
                    jsonrpc: JSONRPC_VERSION,
                    error: Some(RpcError::from_code(RpcErrorCode::ParseError)),
                    id: None,
                };

                return serde_json_core::to_slice(&response, &mut response_json[..]).unwrap();
            }
        };

        let id = request.id;

        if request.jsonrpc != JSONRPC_VERSION {
            let response = RpcResponse {
                jsonrpc: JSONRPC_VERSION,
                error: Some(RpcError::from_code(RpcErrorCode::InvalidRequest)),
                id,
            };

            return serde_json_core::to_slice(&response, &mut response_json[..]).unwrap();
        }

        return match self.handlers.get(request.method) {
            Some(handler) => handler(id, request_json, response_json),
            None => {
                let response = RpcResponse {
                    jsonrpc: JSONRPC_VERSION,
                    error: Some(RpcError::from_code(RpcErrorCode::MethodNotFound)),
                    id,
                };

                serde_json_core::to_slice(&response, &mut response_json[..]).unwrap()
            }
        };
    }

    /// Broadcast a message to all connected clients.
    pub async fn notify<T: Write>(&self, notification_json: &[u8]) -> Result<(), T::Error> {
        let mut notification_json_owned: Vec<u8, MAX_MESSAGE_LEN> = Vec::new();

        notification_json_owned.resize(10, 0).unwrap();
        let varint_size = varint_encode(notification_json.len(), &mut notification_json_owned);
        notification_json_owned.truncate(varint_size);

        notification_json_owned
            .extend_from_slice(notification_json)
            .unwrap();

        let notifications = self.notifications.publisher().unwrap();
        notifications.publish(notification_json_owned).await;

        Ok(())
    }

    /// Serve requests using the given stream.
    pub async fn serve<T: Read + Write>(&self, stream: &mut T) -> Result<(), T::Error> {
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
                    stream.write_all(&notification_json).await?;
                    stream.flush().await?;
                    continue;
                }
                Either::First(WaitResult::Lagged(x)) => {
                    #[cfg(feature = "defmt")]
                    warn!("Dropped {:?} notifications", x);
                }
                Either::Second(Ok(0)) => return Ok(()),
                Either::Second(Ok(n)) => {
                    read_offset += n;

                    // Process complete frames from the buffer
                    while let Some((message_length, varint_len)) =
                        varint_decode(&request_buffer[..read_offset])
                    {
                        if read_offset < varint_len + message_length {
                            // Not enough data for a complete message; wait for more
                            break;
                        }

                        // Extract the full JSON-RPC message
                        let request_json = &request_buffer[varint_len..varint_len + message_length];

                        // Handle the request
                        let response_json_len =
                            self.handle_request(request_json, &mut response_json);

                        // Frame the response using varint
                        let mut length_buf = [0u8; 10];
                        let varint_size = varint_encode(response_json_len, &mut length_buf);

                        // Send the framed response
                        stream.write_all(&length_buf[..varint_size]).await?;
                        stream
                            .write_all(&response_json[..response_json_len])
                            .await?;
                        stream.flush().await?;

                        // Remove processed message from the buffer
                        let remaining = read_offset - (varint_len + message_length);
                        request_buffer.copy_within(varint_len + message_length..read_offset, 0);
                        read_offset = remaining;
                    }
                }
                Either::Second(Err(e)) => return Err(e),
            }
        }
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
        const MAX_CLIENTS: usize = 1;
        const MAX_HANDLERS: usize = 2;
        const MAX_MESSAGE_LEN: usize = 256;

        let mut server: RpcServer<'_, MAX_CLIENTS, MAX_HANDLERS, MAX_MESSAGE_LEN> =
            RpcServer::new();

        server.register_method("echo", |id, _request_json, response_json| {
            let response = RpcResponse {
                jsonrpc: JSONRPC_VERSION,
                id,
                error: None,
            };

            serde_json_core::to_slice(&response, response_json).unwrap()
        });

        let (mut stream1, mut stream2) = MemoryPipe::new();

        tokio::spawn(async move {
            server.serve(&mut stream2).await.unwrap();
        });

        let request = RpcRequest {
            jsonrpc: JSONRPC_VERSION,
            id: Some(1),
            method: "echo",
        };

        let mut request_json = [0u8; 256];
        let request_len = serde_json_core::to_slice(&request, &mut request_json).unwrap();

        // Write the request to the stream
        let mut length_buf = [0u8; 10];
        let varint_size = varint_encode(request_len, &mut length_buf);
        stream1.write_all(&length_buf[..varint_size]).await.unwrap();
        stream1
            .write_all(&request_json[..request_len])
            .await
            .unwrap();

        // Read the response from the stream
        stream1
            .read_exact(&mut length_buf[..varint_size])
            .await
            .unwrap();
        let response_len = varint_decode(&length_buf[..varint_size]).unwrap().0 as usize;
        let mut response_json = vec![0u8; response_len];
        stream1.read_exact(&mut response_json).await.unwrap();

        let (response, _remainder): (RpcResponse, usize) =
            serde_json_core::from_slice(&response_json).unwrap();
        assert_eq!(response.jsonrpc, JSONRPC_VERSION);
        assert_eq!(response.id, Some(1));
        assert!(response.error.is_none());
    }

    #[tokio::test]
    async fn test_notify() {
        const MAX_CLIENTS: usize = 2;
        const MAX_HANDLERS: usize = 2;
        const MAX_MESSAGE_LEN: usize = 256;

        let server = Arc::new(RpcServer::<MAX_CLIENTS, MAX_HANDLERS, MAX_MESSAGE_LEN>::new());

        let server_clone = Arc::clone(&server); // Clone for use in the spawned task
        let (mut stream1, mut stream2) = MemoryPipe::new();

        // Spawn the server task to handle notifications
        tokio::spawn(async move {
            server_clone.serve(&mut stream2).await.unwrap();
        });

        // Sleep to allow the server task to start and subscribe to notifications
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Notification to send
        let notification = RpcResponse {
            jsonrpc: JSONRPC_VERSION,
            id: None,
            error: None,
        };

        let mut notification_json = [0u8; MAX_MESSAGE_LEN];
        let notification_len =
            serde_json_core::to_slice(&notification, &mut notification_json).unwrap();

        // Notify all clients
        server
            .notify::<MemoryPipe>(&notification_json[..notification_len])
            .await
            .unwrap();

        // Read the notification from the stream
        let mut read_length_buf = [0u8; 1];
        stream1.read_exact(&mut read_length_buf[..]).await.unwrap();
        let (received_len, _) = varint_decode(&read_length_buf).unwrap();

        assert_eq!(received_len, notification_len);

        let mut received_notification = vec![0u8; received_len];
        stream1
            .read_exact(&mut received_notification)
            .await
            .unwrap();

        let (response, _): (RpcResponse, usize) =
            serde_json_core::from_slice(&received_notification).unwrap();

        assert_eq!(response.jsonrpc, JSONRPC_VERSION);
        assert_eq!(response.id, None);
        assert!(response.error.is_none());
    }
}
