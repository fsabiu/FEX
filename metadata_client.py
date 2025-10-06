#!/usr/bin/env python3
"""
Metadata WebSocket Client

Connects to the metadata server and displays real-time KLV telemetry
and YOLO detection results.

Usage:
    python3 metadata_client.py --server ws://localhost:8765
"""

import asyncio
import websockets
import json
import logging
import argparse
from datetime import datetime
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MetadataClient")


class MetadataClient:
    """WebSocket client for receiving metadata."""
    
    def __init__(self, server_url, output_file=None, verbose=False):
        """
        Initialize metadata client.
        
        Args:
            server_url: WebSocket server URL (e.g., ws://localhost:8765)
            output_file: Optional file to save metadata JSON
            verbose: Show full JSON output
        """
        self.server_url = server_url
        self.output_file = output_file
        self.verbose = verbose
        
        self.packet_count = 0
        self.start_time = None
        
        if self.output_file:
            self.file_handle = open(self.output_file, 'w')
            self.file_handle.write('[\n')
            logger.info(f"Saving metadata to: {output_file}")
    
    def format_metadata_display(self, metadata):
        """Format metadata for console display."""
        lines = []
        lines.append("\n" + "=" * 80)
        
        # Frame info
        frame = metadata.get('frame', 'N/A')
        timestamp = metadata.get('timestamp', 'N/A')
        lines.append(f"Frame #{frame} | Timestamp: {timestamp}")
        lines.append("=" * 80)
        
        # Telemetry data
        telemetry = metadata.get('telemetry', {})
        if telemetry:
            lines.append("\n📡 TELEMETRY (KLV):")
            
            if 'timestamp_us' in telemetry:
                ts = datetime.fromtimestamp(telemetry['timestamp_us'] / 1_000_000)
                lines.append(f"  Timestamp: {ts.isoformat()}")
            
            if 'latitude' in telemetry and 'longitude' in telemetry:
                lines.append(f"  GPS: {telemetry['latitude']:.7f}°, {telemetry['longitude']:.7f}°")
            
            if 'altitude' in telemetry:
                lines.append(f"  Altitude: {telemetry['altitude']:.2f} m")
            
            if 'heading' in telemetry:
                lines.append(f"  Heading: {telemetry['heading']:.2f}°")
            
            if 'roll' in telemetry:
                lines.append(f"  Roll: {telemetry['roll']:.2f}°")
            
            if 'pitch' in telemetry:
                lines.append(f"  Pitch: {telemetry['pitch']:.2f}°")
        
        # Detection data
        detections = metadata.get('detections', [])
        detection_count = metadata.get('detection_count', 0)
        
        lines.append(f"\n🎯 DETECTIONS: {detection_count} objects")
        
        if detections:
            for i, det in enumerate(detections, 1):
                class_name = det.get('class_name', 'unknown')
                confidence = det.get('confidence', 0.0)
                bbox = det.get('bbox', [])
                
                lines.append(f"  {i}. {class_name.upper()}")
                lines.append(f"     Confidence: {confidence:.2%}")
                if bbox:
                    lines.append(f"     BBox: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")
        
        lines.append("=" * 80)
        
        return '\n'.join(lines)
    
    def format_metadata_compact(self, metadata):
        """Format metadata in compact one-line format."""
        frame = metadata.get('frame', 0)
        telemetry = metadata.get('telemetry', {})
        detection_count = metadata.get('detection_count', 0)
        
        parts = [f"Frame #{frame:06d}"]
        
        # GPS
        if 'latitude' in telemetry and 'longitude' in telemetry:
            parts.append(f"GPS: {telemetry['latitude']:.6f}, {telemetry['longitude']:.6f}")
        
        # Altitude
        if 'altitude' in telemetry:
            parts.append(f"Alt: {telemetry['altitude']:.1f}m")
        
        # Heading
        if 'heading' in telemetry:
            parts.append(f"Hdg: {telemetry['heading']:.1f}°")
        
        # Detections
        parts.append(f"Detections: {detection_count}")
        
        if detection_count > 0:
            detections = metadata.get('detections', [])
            det_names = [f"{d['class_name']}({d['confidence']:.2f})" 
                        for d in detections[:3]]  # Show first 3
            parts.append(f"[{', '.join(det_names)}]")
        
        return " | ".join(parts)
    
    async def connect_and_receive(self):
        """Connect to server and receive metadata."""
        logger.info(f"Connecting to {self.server_url}...")
        
        try:
            async with websockets.connect(self.server_url) as websocket:
                logger.info("✓ Connected to metadata server")
                logger.info("Receiving metadata... (Ctrl+C to stop)\n")
                
                self.start_time = asyncio.get_event_loop().time()
                
                async for message in websocket:
                    try:
                        metadata = json.loads(message)
                        self.packet_count += 1
                        
                        # Display metadata
                        if self.verbose:
                            print(self.format_metadata_display(metadata))
                        else:
                            print(self.format_metadata_compact(metadata))
                        
                        # Save to file if requested
                        if self.output_file:
                            if self.packet_count > 1:
                                self.file_handle.write(',\n')
                            json.dump(metadata, self.file_handle, indent=2)
                            self.file_handle.flush()
                        
                        # Show stats every 100 packets
                        if self.packet_count % 100 == 0:
                            elapsed = asyncio.get_event_loop().time() - self.start_time
                            rate = self.packet_count / elapsed if elapsed > 0 else 0
                            logger.info(f"📊 Received {self.packet_count} packets | Rate: {rate:.2f} pkt/s")
                    
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON: {e}")
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
        
        except websockets.exceptions.WebSocketException as e:
            logger.error(f"WebSocket error: {e}")
            logger.error("Make sure the metadata server is running!")
            return False
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False
        
        return True
    
    def close(self):
        """Close resources."""
        if self.output_file and hasattr(self, 'file_handle'):
            self.file_handle.write('\n]\n')
            self.file_handle.close()
            logger.info(f"✓ Saved {self.packet_count} metadata packets to {self.output_file}")
        
        if self.start_time:
            elapsed = asyncio.get_event_loop().time() - self.start_time
            logger.info(f"\n📊 Summary:")
            logger.info(f"  Total packets: {self.packet_count}")
            logger.info(f"  Duration: {elapsed:.1f}s")
            if elapsed > 0:
                logger.info(f"  Average rate: {self.packet_count / elapsed:.2f} pkt/s")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Metadata WebSocket Client - Display real-time telemetry and detections',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--server',
        default='ws://localhost:8765',
        help='WebSocket server URL'
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Save metadata to JSON file'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show full metadata display (default: compact)'
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
    
    # Create client
    client = MetadataClient(
        server_url=args.server,
        output_file=args.output,
        verbose=args.verbose
    )
    
    try:
        await client.connect_and_receive()
    except KeyboardInterrupt:
        logger.info("\n⚠ Interrupted by user")
    finally:
        client.close()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

