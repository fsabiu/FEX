#!/usr/bin/env python3
"""
Read KLV Metadata from Live Stream

Directly reads and decodes KLV (MISB 0601) metadata from SRT/RTSP/file streams.
No UDP forwarding needed - reads directly from the source.
"""

import av
import sys
import struct
import argparse
import time
from datetime import datetime


def decode_klv_packet(data):
    """
    Simple MISB 0601 KLV packet decoder.
    
    Args:
        data: Raw KLV packet bytes
        
    Returns:
        Dictionary with decoded telemetry, or None if decoding fails
    """
    try:
        # MISB 0601 Universal Key (16 bytes)
        MISB_0601_KEY = bytes([
            0x06, 0x0E, 0x2B, 0x34, 0x02, 0x0B, 0x01, 0x01,
            0x0E, 0x01, 0x03, 0x01, 0x01, 0x00, 0x00, 0x00
        ])
        
        # Check if packet starts with MISB 0601 key
        if not data.startswith(MISB_0601_KEY):
            return None
        
        offset = 16  # Skip key
        
        # Parse BER length
        length_byte = data[offset]
        offset += 1
        
        if length_byte < 128:
            value_length = length_byte
        elif length_byte == 0x81:
            value_length = data[offset]
            offset += 1
        elif length_byte == 0x82:
            value_length = struct.unpack('>H', data[offset:offset+2])[0]
            offset += 2
        else:
            return None
        
        # Parse Local Data Set items
        telemetry = {}
        end_offset = offset + value_length
        
        while offset < end_offset and offset < len(data):
            tag = data[offset]
            offset += 1
            
            if offset >= len(data):
                break
                
            item_length = data[offset]
            offset += 1
            
            if offset + item_length > len(data):
                break
                
            value_bytes = data[offset:offset+item_length]
            offset += item_length
            
            # Decode based on tag
            try:
                if tag == 2:  # Unix timestamp (microseconds)
                    telemetry['timestamp_us'] = struct.unpack('>Q', value_bytes)[0]
                elif tag == 13:  # Sensor latitude
                    scaled = struct.unpack('>i', value_bytes)[0]
                    telemetry['latitude'] = scaled / 1e7
                elif tag == 14:  # Sensor longitude
                    scaled = struct.unpack('>i', value_bytes)[0]
                    telemetry['longitude'] = scaled / 1e7
                elif tag == 15:  # Sensor true altitude
                    scaled = struct.unpack('>H', value_bytes)[0]
                    telemetry['altitude'] = scaled / 10.0
                elif tag == 5:  # Platform roll
                    scaled = struct.unpack('>h', value_bytes)[0]
                    telemetry['roll'] = scaled / 100.0
                elif tag == 6:  # Platform pitch
                    scaled = struct.unpack('>h', value_bytes)[0]
                    telemetry['pitch'] = scaled / 100.0
                elif tag == 7:  # Platform heading
                    scaled = struct.unpack('>H', value_bytes)[0]
                    telemetry['heading'] = scaled / 100.0
            except struct.error:
                # Skip malformed fields
                continue
        
        return telemetry
        
    except Exception as e:
        return None


def read_klv_from_stream(source_url, duration=60, verbose=False):
    """
    Read and decode KLV metadata from a live stream.
    
    Args:
        source_url: SRT/RTSP/file URL to read from
        duration: How long to read in seconds (0 = indefinite)
        verbose: Show raw packet data
    """
    print("=" * 70)
    print("KLV Metadata Reader")
    print(f"Source: {source_url}")
    print(f"Duration: {'indefinite' if duration == 0 else f'{duration} seconds'}")
    print("=" * 70)
    print()
    
    packet_count = 0
    decoded_count = 0
    start_time = time.time()
    last_packet_time = start_time
    
    try:
        print(f"Opening stream: {source_url}")
        
        # Open stream with appropriate options
        options = {}
        if source_url.startswith('rtsp://'):
            options['rtsp_transport'] = 'tcp'
        options['timeout'] = '10000000'  # 10 second timeout
        
        container = av.open(source_url, options=options)
        
        # Find data stream (KLV)
        data_stream = None
        print(f"\nAvailable streams:")
        for i, stream in enumerate(container.streams):
            print(f"  Stream {i}: {stream.type} - {stream}")
            if stream.type == 'data':
                data_stream = stream
        
        if not data_stream:
            print("\n✗ ERROR: No data stream found in source")
            print("  Make sure your stream contains KLV metadata (mapped with -map 0:d)")
            return 1
        
        print(f"\n✓ Using data stream for KLV")
        print(f"✓ Stream opened successfully")
        print("Reading KLV packets... (Ctrl+C to stop)")
        print()
        
        # Process packets
        for packet in container.demux(data_stream):
            # Check duration limit
            if duration > 0 and (time.time() - start_time) >= duration:
                print(f"\n✓ Duration limit reached ({duration}s)")
                break
            
            if packet.size == 0:
                continue
            
            packet_count += 1
            current_time = time.time()
            time_diff = current_time - last_packet_time
            last_packet_time = current_time
            
            # Extract raw packet data
            packet_data = bytes(packet)
            
            print(f"\n{'='*70}")
            print(f"Packet #{packet_count} | Size: {len(packet_data)} bytes | "
                  f"Time since last: {time_diff:.3f}s")
            print(f"{'='*70}")
            
            if verbose:
                print(f"Raw data (first 100 bytes): {packet_data[:100].hex()}")
            
            # Try to decode KLV packet
            telemetry = decode_klv_packet(packet_data)
            
            if telemetry:
                decoded_count += 1
                print("\n📦 KLV Packet Decoded:")
                print(f"  MISB 0601 Universal Key detected")
                
                print("\n📍 MISB 0601 Telemetry Data:")
                
                # Extract and display telemetry fields
                if 'timestamp_us' in telemetry:
                    ts = datetime.fromtimestamp(telemetry['timestamp_us'] / 1_000_000)
                    print(f"  Timestamp: {ts.isoformat()} UTC")
                
                if 'latitude' in telemetry:
                    print(f"  Latitude: {telemetry['latitude']:.7f}°")
                
                if 'longitude' in telemetry:
                    print(f"  Longitude: {telemetry['longitude']:.7f}°")
                
                if 'altitude' in telemetry:
                    print(f"  Altitude: {telemetry['altitude']:.2f} m")
                
                if 'roll' in telemetry:
                    print(f"  Roll: {telemetry['roll']:.2f}°")
                
                if 'pitch' in telemetry:
                    print(f"  Pitch: {telemetry['pitch']:.2f}°")
                
                if 'heading' in telemetry:
                    print(f"  Heading: {telemetry['heading']:.2f}°")
                
                print(f"\n✓ Packet successfully decoded")
            else:
                print(f"\n⚠ Could not decode KLV packet")
                print(f"Raw data (first 100 bytes): {packet_data[:100].hex()}")
                
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        elapsed = time.time() - start_time
        
        print("\n" + "=" * 70)
        print("📊 Summary:")
        print(f"  Total packets received: {packet_count}")
        print(f"  Successfully decoded: {decoded_count}")
        print(f"  Duration: {elapsed:.1f}s")
        if packet_count > 0:
            print(f"  Average rate: {packet_count / elapsed:.2f} packets/sec")
            print(f"  Decode success rate: {decoded_count/packet_count*100:.1f}%")
        print("=" * 70)
    
    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Read and decode KLV metadata directly from SRT/RTSP/file streams',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'source',
        help='Source URL (e.g., srt://host:port, rtsp://host/path, or file.ts)'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=60,
        help='Duration to read in seconds (0 = indefinite)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show raw packet data'
    )
    
    args = parser.parse_args()
    
    return read_klv_from_stream(
        source_url=args.source,
        duration=args.duration,
        verbose=args.verbose
    )


if __name__ == '__main__':
    sys.exit(main())

