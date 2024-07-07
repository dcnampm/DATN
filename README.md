1. create env
requires: python=3.9
2. install 
pip install -r requirements.txt
pip install -e .
3. Test model yolov10s_custom
python app.py
# some image test: folder image_test
4. Test model yolov10 + camera
python camera_yolov10.py