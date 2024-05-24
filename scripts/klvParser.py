import sys
import klvdata
import subprocess
import time

def extract_klv_data(rtsp_url):
    # Run FFmpeg command to extract KLV data from the RTSP stream
    command = [
        'ffmpeg',
        '-i', rtsp_url,
        '-map', '0:1',  # Assuming the KLV data is on the second stream
        '-c', 'copy',
        '-f', 'data',
        '-'
    ]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    elapsed_time = 0
    # Read and parse KLV data from the stdout
    while True:
        klv_packet = process.stdout.read(1024)
        if not klv_packet:
            time.sleep(1)
            print("waiting for stream...")
            continue
        
        try:
            klv_data = klvdata.StreamParser(klv_packet)
            for packet in klv_data:
                # print(packet)
                    metadata = packet.MetadataList()
                    for key, value in metadata.items():
                        print(key, value)
        except Exception as e:
            continue
            print(f"Failed to parse KLV packet: {e}")

    process.stdout.close()
    process.wait()

if __name__ == "__main__":
    rtsp_url = "rtsp://10.8.0.1:8554/live/UAS-BIXBY"
    extract_klv_data(rtsp_url)
