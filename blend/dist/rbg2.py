from http.server import BaseHTTPRequestHandler, HTTPServer
from PIL import Image
from rembg import remove
import io


class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/api/remove":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            # Simplified image extraction (assumes single file upload)
            image_data = post_data.split(b"\r\n\r\n")[1]
            image_data = image_data[:-2]

            try:
                input_image = Image.open(io.BytesIO(image_data))
                output_image = remove(input_image)

                self.send_response(200)
                self.send_header("Content-type", "image/png")
                self.end_headers()
                output_image.save(self.wfile, "PNG")

            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(f"Error: {e}".encode())
        else:
            self.send_response(404)
            self.end_headers()


httpd = HTTPServer(("localhost", 7000), SimpleHTTPRequestHandler)
print("Server running on http://localhost:7000")
httpd.serve_forever()