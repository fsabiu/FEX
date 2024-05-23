import asyncio
import xml.etree.ElementTree as ET
import pytak
import random
from configparser import ConfigParser
import time

def object2COT(object) :
    cot_types = {
        "pedestrian": "a-h-G",
        "car": "a-f-G-E-V-C",
        "van": "a-f-G-E-V-V",
        "truck": "a-f-G-E-V-T",
        "military_tank": "a-f-G-U-C-T",
        "military_truck": "a-f-G-U-L-T",
        "military_vehicle": "a-f-G-U-C-V",
        "BMP-1": "a-f-G-U-C-V-B",
        "Rosomak": "a-f-G-U-C-V-R",
        "T72": "a-f-G-U-C-T-T72",
        "people": "a-h-G",
        "soldier": "a-f-G-U-C-I",
        "trench": "a-f-G-U-C-F-T",
        "hidden_object": "a-f-G-E-O-H"
    }    
    return cot_types[object]

def gen_cot_detection(object,lat,lon,h):
    """Generate CoT Event."""
    root = ET.Element("event")
    root.set("version", "2.0")
    root.set("type", object2COT(object))  # insert your type of marker
    root.set("uid", object)
    root.set("how", "a")
    root.set("time", pytak.cot_time())
    root.set("start", pytak.cot_time())
    root.set(
        "stale", pytak.cot_time(60)
    )  # time difference in seconds from 'start' when stale initiates

    pt_attr = {
        "lat": str(lat),  
        "lon": str(lon),  
        "hae": str(h),
        "ce": "",
        "le": "",
    }

    ET.SubElement(root, "point", attrib=pt_attr)

    return ET.tostring(root)

class MySender(pytak.QueueWorker):
    """
    Defines how you process or generate your Cursor-On-Target Events.
    From there it adds the COT Events to a queue for TX to a COT_URL.
    """

    def __init__(self, tx_queue, config, object, lat, lon, h):
        super().__init__(tx_queue, config)
        self.object = object
        self.lat = lat
        self.lon = lon
        self.h = h
        
    async def handle_data(self, data):
        """Handle pre-CoT data, serialize to CoT Event, then puts on queue."""
        event = data
        await self.put_queue(event)
        
    async def run(self):
        """Run the loop for processing or generating pre-CoT data."""
        data = gen_cot_detection(self.object, self.lat, self.lon, self.h)
        self._logger.info("Sending:\n%s\n", data.decode())
        await self.handle_data(data)

def getConfig() :
    config = ConfigParser()
    config["mycottool"] = {"COT_URL": "tls://10.8.0.1:8089", 
                           "PYTAK_TLS_CLIENT_CERT": "/home/ubuntu/shared/tak_certs/oracle1.p12",
                           "PYTAK_TLS_CLIENT_PASSWORD" : "atakatak",
                           "PYTAK_TLS_DONT_VERIFY" : 1,
                            "PYTAK_TLS_DONT_CHECK_HOSTNAME": 1
                           } 
    config = config["mycottool"]
    return config

async def main():
    """Main definition of your program, sets config params and
    adds your serializer to the asyncio task list.
    """
    config = getConfig()

    # Initializes worker queues and tasks.
    clitool = pytak.CLITool(config)
    await clitool.setup()
    print("done")
# Generate test data
    while True:
        # Generate a random set of test data
        object_type = random.choice(["pedestrian", "car", "van", "truck", "military_tank", "military_truck", "military_vehicle", "BMP-1", "Rosomak", "T72", "people", "soldier", "trench", "hidden_object"])
        lat = round(random.uniform(-90, 90), 6)
        lon = round(random.uniform(-180, 180), 6)
        h = round(random.uniform(-100, 100), 2)
        
        sender = MySender(clitool.tx_queue, config, object_type, lat, lon, h)
        clitool.add_tasks(set([sender]))
        

    # Start all tasks.
        await clitool.run()
        await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(main())