#!/usr/bin/env python3
"""
Metadata WebSocket Server

This server runs alongside srt_yolo_hls_inference.py and broadcasts
real-time metadata (KLV telemetry + detections) to connected clients.

Usage:
    python3 metadata_server.py --port 8765
"""

import asyncio
import websockets
import json
import logging
import argparse
import socket
import threading
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MetadataServer")


class MetadataServer:
    """WebSocket server for broadcasting metadata to clients."""
    
    def __init__(self, host='0.0.0.0', port=8765, udp_port=5555):
        """
        Initialize metadata server.
        
        Args:
            host: WebSocket server host
            port: WebSocket server port
            udp_port: UDP port to receive metadata from inference script
        """
        self.host = host
        self.port = port
        self.udp_port = udp_port
        
        self.clients = set()
        self.latest_metadata = None
        self.metadata_count = 0
        
        self.udp_socket = None
        self.running = False
    
    async def register_client(self, websocket):
        """Register a new WebSocket client."""
        self.clients.add(websocket)
        logger.info(f"Client connected: {websocket.remote_address}. Total clients: {len(self.clients)}")
        
        # Send latest metadata immediately if available
        if self.latest_metadata:
            try:
                await websocket.send(json.dumps(self.latest_metadata))
            except Exception as e:
                logger.error(f"Error sending initial metadata: {e}")
    
    async def unregister_client(self, websocket):
        """Unregister a WebSocket client."""
        self.clients.discard(websocket)
        logger.info(f"Client disconnected: {websocket.remote_address}. Total clients: {len(self.clients)}")
    
    async def broadcast_metadata(self, metadata):
        """Broadcast metadata to all connected clients."""
        if not self.clients:
            return
        
        message = json.dumps(metadata)
        disconnected_clients = set()
        
        for client in self.clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
            except Exception as e:
                logger.error(f"Error sending to client: {e}")
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        for client in disconnected_clients:
            await self.unregister_client(client)
    
    async def handle_client(self, websocket, path):
        """Handle WebSocket client connection."""
        await self.register_client(websocket)
        
        try:
            # Keep connection alive
            async for message in websocket:
                # Echo back or handle client messages if needed
                logger.debug(f"Received from client: {message}")
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister_client(websocket)
    
    def udp_receiver_thread(self):
        """Thread to receive metadata via UDP."""
        logger.info(f"Starting UDP receiver on port {self.udp_port}")
        
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.bind(('0.0.0.0', self.udp_port))
        self.udp_socket.settimeout(1.0)
        
        while self.running:
            try:
                data, addr = self.udp_socket.recvfrom(65535)
                metadata = json.loads(data.decode('utf-8'))
                
                self.latest_metadata = metadata
                self.metadata_count += 1
                
                # Schedule broadcast in asyncio loop
                asyncio.run_coroutine_threadsafe(
                    self.broadcast_metadata(metadata),
                    self.loop
                )
                
                if self.metadata_count % 100 == 0:
                    logger.info(f"Received {self.metadata_count} metadata packets")
                    
            except socket.timeout:
                continue
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON: {e}")
            except Exception as e:
                logger.error(f"UDP receiver error: {e}")
        
        self.udp_socket.close()
        logger.info("UDP receiver stopped")
    
    async def start(self):
        """Start the WebSocket server."""
        self.running = True
        self.loop = asyncio.get_event_loop()
        
        # Start UDP receiver in separate thread
        udp_thread = threading.Thread(target=self.udp_receiver_thread, daemon=True)
        udp_thread.start()
        
        # Start WebSocket server
        logger.info(f"Starting WebSocket server on ws://{self.host}:{self.port}")
        logger.info(f"Listening for metadata on UDP port {self.udp_port}")
        logger.info("Waiting for connections...")
        
        async with websockets.serve(self.handle_client, self.host, self.port):
            await asyncio.Future()  # Run forever
    
    def stop(self):
        """Stop the server."""
        logger.info("Stopping metadata server...")
        self.running = False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Metadata WebSocket Server - Broadcast real-time metadata to clients',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='WebSocket server host (0.0.0.0 = all interfaces)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8765,
        help='WebSocket server port'
    )
    parser.add_argument(
        '--udp-port',
        type=int,
        default=5555,
        help='UDP port to receive metadata from inference script'
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Create and start server
    server = MetadataServer(
        host=args.host,
        port=args.port,
        udp_port=args.udp_port
    )
    
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logger.info("\nStopping server...")
        server.stop()


if __name__ == '__main__':
    main()

