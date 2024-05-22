#!/usr/bin/env python3

import asyncio
import xml.etree.ElementTree as ET
import pytak

from configparser import ConfigParser

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
        "lat": lat,  
        "lon": lon,  
        "hae": h,
        "ce": "",
        "le": "",
    }

    ET.SubElement(root, "point", attrib=pt_attr)

    return ET.tostring(root)

def gen_cot():
    """Generate CoT Event."""
    root = ET.Element("event")
    root.set("version", "2.0")
    root.set("type", "a-h-A-M-A")  # insert your type of marker
    root.set("uid", "name_your_marker")
    root.set("how", "m-g")
    root.set("time", pytak.cot_time())
    root.set("start", pytak.cot_time())
    root.set(
        "stale", pytak.cot_time(60)
    )  # time difference in seconds from 'start' when stale initiates

    pt_attr = {
        "lat": "40.781789",  # set your lat (this loc points to Central Park NY)
        "lon": "-73.968698",  # set your long (this loc points to Central Park NY)
        "hae": "0",
        "ce": "10",
        "le": "10",
    }

    ET.SubElement(root, "point", attrib=pt_attr)

    return ET.tostring(root)


class MySender(pytak.QueueWorker):
    """
    Defines how you process or generate your Cursor-On-Target Events.
    From there it adds the COT Events to a queue for TX to a COT_URL.
    """

    async def handle_data(self, data):
        """Handle pre-CoT data, serialize to CoT Event, then puts on queue."""
        event = data
        await self.put_queue(event)

    async def run(self, number_of_iterations=-1):
        """Run the loop for processing or generating pre-CoT data."""
        while 1:
            data = gen_cot()
            self._logger.info("Sending:\n%s\n", data.decode())
            await self.handle_data(data)
            await asyncio.sleep(5)


class MyReceiver(pytak.QueueWorker):
    """Defines how you will handle events from RX Queue."""

    async def handle_data(self, data):
        """Handle data from the receive queue."""
        self._logger.info("Received:\n%s\n", data.decode())

    async def run(self):  # pylint: disable=arguments-differ
        """Read from the receive queue, put data onto handler."""
        while 1:
            data = (
                await self.queue.get()
            )  # this is how we get the received CoT from rx_queue
            await self.handle_data(data)


async def main():
    """Main definition of your program, sets config params and
    adds your serializer to the asyncio task list.
    """
    config = ConfigParser()
    config["mycottool"] = {"COT_URL": "tls://10.8.0.1:8089", 
                           "PYTAK_TLS_CLIENT_CERT": "/home/ubuntu/shared/tak_certs/oracle1.p12",
                           "PYTAK_TLS_CLIENT_PASSWORD" : "atakatak",
                           "PYTAK_TLS_DONT_VERIFY" : 1,
                            "PYTAK_TLS_DONT_CHECK_HOSTNAME": 1
                           }    
    config = config["mycottool"]

    # Initializes worker queues and tasks.
    clitool = pytak.CLITool(config)
    await clitool.setup()

    # Add your serializer to the asyncio task list.
    clitool.add_tasks(
        set([MySender(clitool.tx_queue, config), MyReceiver(clitool.rx_queue, config)])
    )

    # Start all tasks.
    await clitool.run()


if __name__ == "__main__":
    asyncio.run(main())