import os
import time
import urllib.request

from flask import Flask, request
from flask_restful import reqparse, Api, Resource

from jsbcl_pdf_extractor import TableExtractor

app = Flask(__name__)
api = Api(app)

HOME_DIRECTORY = os.path.dirname(os.path.abspath(__file__))


class ExtractDataAPI(Resource):
    def post(self):
        start = time.time()

        if "file" not in request.files:
            return {"error": "Missing file in request"}, 400

        file = request.files["file"]
        if file.filename == "":
            return {"error": "Empty filename"}, 400

        pdf_folder = os.path.join(HOME_DIRECTORY, "pdf_folder")
        os.makedirs(pdf_folder, exist_ok=True)
        pdf_file_path = os.path.join(pdf_folder, "original_pdf.pdf")

        try:
            file.save(pdf_file_path)
        except Exception as e:
            return {"error": f"Failed to save uploaded file: {str(e)}"}, 500

        try:
            extractor = TableExtractor(pdf_path=pdf_file_path)
            result = extractor.extract_stock_purchase_data(
                extraction_time_seconds=round(time.time() - start, 2)
            )
        except Exception as e:
            return {"error": f"Extraction failed: {str(e)}"}, 500

        return result, 200

    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument("file_path", location="args", required=True)
        args = parser.parse_args()

        file_path_or_url = args["file_path"]
        start = time.time()

        pdf_folder = os.path.join(HOME_DIRECTORY, "pdf_folder")
        os.makedirs(pdf_folder, exist_ok=True)
        pdf_file_path = os.path.join(pdf_folder, "original_pdf.pdf")

        try:
            if file_path_or_url.startswith("http://") or file_path_or_url.startswith("https://"):
                resp = urllib.request.urlopen(file_path_or_url)
                with open(pdf_file_path, "wb") as f:
                    f.write(resp.read())
            else:
                local_path = os.path.abspath(file_path_or_url)

                if not os.path.exists(local_path):
                    return {"error": f"Local file not found: {local_path}"}, 404

                with open(local_path, "rb") as src, open(pdf_file_path, "wb") as dest:
                    dest.write(src.read())

        except Exception as e:
            return {"error": f"Failed to load PDF: {str(e)}"}, 400

        try:
            extractor = TableExtractor(pdf_path=pdf_file_path)
            result = extractor.extract_stock_purchase_data(
                extraction_time_seconds=round(time.time() - start, 2)
            )
        except Exception as e:
            return {"error": f"Extraction failed: {str(e)}"}, 500

        return result, 200


api.add_resource(ExtractDataAPI, "/extract_data")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8089)
