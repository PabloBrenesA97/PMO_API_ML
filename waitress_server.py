from waitress import serve
import api_pmo
serve(api_pmo.app, host='0.0.0.0', port=5000)