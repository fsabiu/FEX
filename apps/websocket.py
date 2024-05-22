import asyncio
import json
import websockets
import geojson
from geojson import Point, Feature, FeatureCollection

# Inicializar la lista de features
features = []

async def handle_message(websocket):
    async for message in websocket:
        data = json.loads(message)

        if data.get("action") == "add_point":
            # Agregar o actualizar un punto
            lat = data.get("latitude")
            lon = data.get("longitude")
            properties = data.get("properties", {})
            feature_id = properties.get("id")
            
            if lat is not None and lon is not None and feature_id is not None:
                # Buscar si ya existe un feature con el mismo id
                feature_exists = False
                for feature in features:
                    if feature["properties"].get("id") == feature_id:
                        # Actualizar las coordenadas del punto existente
                        feature["geometry"]["coordinates"] = [lon, lat]
                        feature_exists = True
                        break

                if not feature_exists:
                    # Agregar un nuevo punto si no existe uno con el mismo id
                    point = Point((lon, lat))
                    feature = Feature(geometry=point, properties=properties)
                    features.append(feature)

                # Enviar la colección de features actualizada
                feature_collection = FeatureCollection(features)
                await websocket.send(geojson.dumps(feature_collection))
            else:
                await websocket.send(json.dumps({"status": "error", "message": "Invalid coordinates or id"}))

        elif data.get("action") == "get_geojson":
            # Crear FeatureCollection y devolverla
            feature_collection = FeatureCollection(features)
            await websocket.send(geojson.dumps(feature_collection))

async def main():
    async with websockets.serve(handle_message, "0.0.0.0", 12001):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
