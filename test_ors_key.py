import requests

def validate_ors_key(api_key):
    url = "https://api.openrouteservice.org/v2/directions/driving-car"
    headers = {"Authorization": api_key}
    params = {
        "coordinates": [[8.34234, 48.23424], [8.34423, 48.26424]],
        "profile": "driving-car",
        "format": "json"
    }
    try:
        response = requests.post(url, json=params, headers=headers)
        if response.status_code == 200:
            print("Clave ORS válida.")
            return True
        else:
            print(f"Error: {response.status_code} - {response.json().get('error', 'No details')}")
            return False
    except Exception as e:
        print(f"Excepción: {str(e)}")
        return False

if __name__ == "__main__":
    api_key = "5b3ce3597851110001cf6248c8e57a115b0dc3314d3cdedbc38516f97ac457a2c3bdac9542cd1ec7"

    validate_ors_key(api_key)