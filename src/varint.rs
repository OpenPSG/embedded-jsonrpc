//! # Varint Encoding and Decoding Library
//!
//! This crate provides utilities for encoding and decoding unsigned integers (`usize`)
//! using the variable-length integer (varint) format. Varints are commonly used in protocols
//! and file formats like Protocol Buffers to efficiently encode integers of varying sizes.
//!
//! ## Functionality
//!
//! - [`encode`] - Encodes a `usize` value into a buffer as a varint and returns the number of bytes written.
//! - [`decode`] - Decodes a varint from a byte buffer, returning the value and the number of bytes read.
//!
//! ## Examples
//!
//! ### Encoding a Value
//!
//! ```rust
//! use embedded_jsonrpc::varint::encode;
//!
//! let mut buffer = [0u8; 10]; // Create a buffer with sufficient size
//! let bytes_written = encode(300, &mut buffer);
//! println!("Encoded 300 into {:?} using {} bytes", &buffer[..bytes_written], bytes_written);
//! ```
//!
//! ### Decoding a Value
//!
//! ```rust
//! use embedded_jsonrpc::varint::decode;
//!
//! let buffer = [0xac, 0x02]; // Encoded varint for 300
//! if let Some((value, bytes_read)) = decode(&buffer) {
//!     println!("Decoded value: {}, bytes read: {}", value, bytes_read);
//! } else {
//!     println!("Failed to decode value");
//! }
//! ```
//!
//! ## License
//!
//! This crate is licensed under the Mozilla Public License 2.0 (MPL-2.0).
//! See the LICENSE file for more details.
//!

/// Encodes a usize into a varint and writes it into the provided buffer.
/// Returns the number of bytes written.
pub fn encode(value: usize, buf: &mut [u8]) -> usize {
    let mut value = value;
    let mut i = 0;

    while value >= 0x80 {
        buf[i] = (value as u8 & 0x7F) | 0x80; // Lower 7 bits + MSB = 1
        value >>= 7; // Shift out the lower 7 bits
        i += 1;
    }

    buf[i] = value as u8; // Write the final byte with MSB = 0
    i + 1
}

/// Decodes a varint from the provided buffer.
/// Returns the decoded value and the number of bytes read, or `None` if the buffer is incomplete.
pub fn decode(buf: &[u8]) -> Option<(usize, usize)> {
    let mut value = 0usize;
    let mut shift = 0;
    let mut i = 0;

    for byte in buf {
        let byte = *byte as usize;
        value |= (byte & 0x7F) << shift; // Add the lower 7 bits to the value
        shift += 7;

        i += 1;
        if byte & 0x80 == 0 {
            // MSB = 0 indicates the end of the varint
            return Some((value, i));
        }

        if shift >= usize::BITS as usize {
            // Varint is too long; invalid
            return None;
        }
    }

    None // Buffer ended before varint was fully decoded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_known_vectors() {
        // These arrays are of the form (value, expected_encoded_form).
        let vectors: [(usize, [u8; 10], usize); 9] = [
            (0, [0x00, 0, 0, 0, 0, 0, 0, 0, 0, 0], 1),
            (1, [0x01, 0, 0, 0, 0, 0, 0, 0, 0, 0], 1),
            (127, [0x7f, 0, 0, 0, 0, 0, 0, 0, 0, 0], 1),
            (128, [0x80, 0x01, 0, 0, 0, 0, 0, 0, 0, 0], 2),
            (255, [0xff, 0x01, 0, 0, 0, 0, 0, 0, 0, 0], 2),
            (300, [0xac, 0x02, 0, 0, 0, 0, 0, 0, 0, 0], 2),
            (16384, [0x80, 0x80, 0x01, 0, 0, 0, 0, 0, 0, 0], 3),
            (2097151, [0xff, 0xff, 0x7f, 0, 0, 0, 0, 0, 0, 0], 3),
            (268435455, [0xff, 0xff, 0xff, 0x7f, 0, 0, 0, 0, 0, 0], 4),
        ];

        for (val, expected, expected_size) in vectors.iter() {
            let mut buf = [0u8; 10]; // Buffer larger than necessary to test encoding
            let encoded_size = encode(*val, &mut buf);
            assert_eq!(
                &buf[..encoded_size],
                &expected[..*expected_size],
                "Failed encoding for value {}",
                val
            );

            let decoded = decode(&buf[..encoded_size]).unwrap();
            assert_eq!(
                decoded,
                (*val, *expected_size),
                "Failed decoding for value {}",
                val
            );
        }
    }

    #[test]
    fn test_encoding_decoding_round_trip() {
        let mut buf = [0u8; 10];
        let values = [
            0,
            1,
            127,
            128,
            255,
            1023,
            16383,
            16384,
            2097151,
            268435455,
            usize::MAX,
        ];

        for &val in &values {
            let encoded_size = encode(val, &mut buf);
            let decoded = decode(&buf[..encoded_size]).unwrap();
            assert_eq!(
                decoded,
                (val, encoded_size),
                "Failed round-trip for value {}",
                val
            );
        }
    }

    #[test]
    fn test_incomplete_buffer() {
        let mut buf = [0u8; 2];
        encode(300, &mut buf);
        assert_eq!(
            decode(&buf[..1]),
            None,
            "Should have failed due to incomplete buffer"
        ); // Incomplete varint
    }

    #[test]
    fn test_invalid_varint_too_long() {
        let mut buf = [0x80u8; 11]; // All continuation bits set
        buf[10] = 0; // Terminate varint
        assert_eq!(
            decode(&buf),
            None,
            "Should have failed due to varint being too long"
        ); // More than 10 bytes, thus more than usize bits, invalid for usize
    }
}
