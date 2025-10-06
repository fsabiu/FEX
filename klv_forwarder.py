#!/usr/bin/env python3
"""
KLV Forwarder - Extract KLV data from SRT/RTSP stream and forward via UDP

This script extracts KLV (MISB 0601) data packets from a video stream
and forwards them via UDP to test_klv_receiver.py or other KLV consumers.
"""

import av
import socket
import sys
import argparse
import time
from datetime import datetime


def forward_klv(source_url, target_host='127.0.0.1', target_port=12345, duration=0):
    """
    Extract KLV data from stream and forward via UDP.
    
    Args:
        source_url: SRT/RTSP/file URL to read from
        target_host: UDP destination host
        target_port: UDP destination port
        duration: How long to run in seconds (0 = indefinite)
    """
    print("=" * 70)
    print("KLV Forwarder")
    print(f"Source: {source_url}")
    print(f"Target: {target_host}:{target_port}")
    print(f"Duration: {'indefinite' if duration == 0 else f'{duration} seconds'}")
    print("=" * 70)
    print()
    
    # Create UDP socket for sending
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    packet_count = 0
    start_time = time.time()
    klv_fix = b''
    
    try:
        print(f"Opening stream: {source_url}")
        container = av.open(source_url, options={
            'rtsp_transport': 'tcp',  # Use TCP for RTSP
            'timeout': '10000000',     # 10 second timeout
        })
        
        # Find data stream (KLV)
        data_stream = None
        for stream in container.streams:
            if stream.type == 'data':
                data_stream = stream
                print(f"✓ Found data stream: {stream}")
                break
        
        if not data_stream:
            print("✗ ERROR: No data stream found in source")
            print("  Make sure your stream contains KLV metadata")
            return 1
        
        print(f"✓ Stream opened successfully")
        print(f"Forwarding KLV packets to {target_host}:{target_port}...")
        print()
        
        # Process packets
        for packet in container.demux(data_stream):
            # Check duration limit
            if duration > 0 and (time.time() - start_time) >= duration:
                print(f"\n✓ Duration limit reached ({duration}s)")
                break
            
            if packet.size == 0:
                continue
            
            # Extract raw packet data
            packet_data = bytes(packet)
            
            if len(packet_data) > 0:
                # Send via UDP
                sock.sendto(packet_data, (target_host, target_port))
                packet_count += 1
                
                current_time = time.time()
                elapsed = current_time - start_time
                
                print(f"[{elapsed:7.2f}s] Packet #{packet_count:04d} | "
                      f"Size: {len(packet_data):5d} bytes | "
                      f"Sent to {target_host}:{target_port}")
                
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        elapsed = time.time() - start_time
        sock.close()
        
        print("\n" + "=" * 70)
        print("📊 Summary:")
        print(f"  Total KLV packets forwarded: {packet_count}")
        print(f"  Duration: {elapsed:.1f}s")
        if packet_count > 0:
            print(f"  Average rate: {packet_count / elapsed:.2f} packets/sec")
        print("=" * 70)
    
    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='KLV Forwarder - Extract KLV from stream and forward via UDP',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'source',
        help='Source URL (SRT/RTSP/file) containing KLV data'
    )
    parser.add_argument(
        '--host',
        default='127.0.0.1',
        help='Target UDP host'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=12345,
        help='Target UDP port'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=0,
        help='Duration to run in seconds (0 = indefinite)'
    )
    
    args = parser.parse_args()
    
    return forward_klv(
        source_url=args.source,
        target_host=args.host,
        target_port=args.port,
        duration=args.duration
    )


if __name__ == '__main__':
    sys.exit(main())

