use embedded_io_async::{ErrorType, Read, Write};
use std::sync::Arc;
use tokio::sync::{
    mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender},
    watch, Mutex,
};

/// A simple in-memory pipe for testing
pub struct MemoryPipe {
    read_rx: Arc<Mutex<UnboundedReceiver<u8>>>,
    write_tx: Arc<Mutex<UnboundedSender<u8>>>,
    drop_signal: Arc<watch::Sender<()>>,
    drop_notifier: watch::Receiver<()>,
}

impl MemoryPipe {
    pub fn new() -> (Self, Self) {
        let (tx1, rx1) = unbounded_channel();
        let (tx2, rx2) = unbounded_channel();
        let (drop_signal1, drop_notifier1) = watch::channel(());
        let (drop_signal2, drop_notifier2) = watch::channel(());

        let stream1 = MemoryPipe {
            read_rx: Arc::new(Mutex::new(rx1)),
            write_tx: Arc::new(Mutex::new(tx2)),
            drop_signal: Arc::new(drop_signal1),
            drop_notifier: drop_notifier2.clone(),
        };

        let stream2 = MemoryPipe {
            read_rx: Arc::new(Mutex::new(rx2)),
            write_tx: Arc::new(Mutex::new(tx1)),
            drop_signal: Arc::new(drop_signal2),
            drop_notifier: drop_notifier1.clone(),
        };

        (stream1, stream2)
    }
}

impl Drop for MemoryPipe {
    fn drop(&mut self) {
        // Notify the paired pipe to cancel pending operations
        let _ = self.drop_signal.send(());
    }
}

impl ErrorType for MemoryPipe {
    type Error = embedded_io_async::ErrorKind;
}

impl Read for MemoryPipe {
    async fn read(&mut self, buf: &mut [u8]) -> Result<usize, Self::Error> {
        let mut rx = self.read_rx.lock().await;
        let mut bytes_read = 0;

        for byte in buf.iter_mut() {
            tokio::select! {
                // Attempt to read data from the channel
                data = rx.recv() => {
                    if let Some(data) = data {
                        *byte = data;
                        bytes_read += 1;
                    } else {
                        // Channel is closed; no more data
                        break;
                    }
                }
                _ = self.drop_notifier.changed() => {
                    return Err(embedded_io_async::ErrorKind::Other);
                }
            }

            // Stop if the channel is empty
            if rx.is_empty() {
                break;
            }
        }

        Ok(bytes_read)
    }
}

impl Write for MemoryPipe {
    async fn write(&mut self, buf: &[u8]) -> Result<usize, Self::Error> {
        let tx = self.write_tx.lock().await;

        for byte in buf {
            if self.drop_notifier.has_changed().unwrap_or(false) {
                return Err(embedded_io_async::ErrorKind::WriteZero);
            }

            tx.send(*byte)
                .map_err(|_| embedded_io_async::ErrorKind::WriteZero)?;
        }

        Ok(buf.len())
    }
}
