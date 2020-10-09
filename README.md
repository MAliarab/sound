<div dir='rtl' align='right'>
	
# ماژول شناسایی پهپاد در تصویر


مدل پیاده سازی شده برای شناسایی پهپاد در تصویر و رسم bbox آموزش دیده است و قابلیت اجرا بر روی cpu و gpu و multi-gpu را دارد.


## معماری مدل

مدل پیاده سازی شده بر اساس یادگیری عمیق و با استفاده از معماری  r-cnn و با شبکه ی keras-retinanet پیاده سازی شده است.

## شناسایی

### نصب و راه اندازی مدل


پیش نیازهای پروژه و ابزارهای لازم در فایل requirement.txt موجود است و با استفاده از دستور زیر میتوان آنها را نصب کرد.
`pip install -r requirements.txt`
#### استفاده از مدل پیش آموزش دیده

برای شناسایی نیاز است مدل آموزش دیده در فولدر Trained-Model قرار داده شود که میتوان از لینک زیر مدل آموزش دیده را دانلود کرد و در این فولدر قرار داد. (فایل drone-detection-v5.h5)
https://drive.google.com/file/d/1u4EvtOl8PAOkj6Rirk16x0a9rQ8S8k6o/view?usp=sharing
در صورت آموزش مجدد مدل باید فایل جدید با این فایل جایگزین شود


#### اجرا روی GPU
نکته: برای اجرای مدل بر روی gpu باید در فایل requirement.txt خط tensorflow به tensorflow-gpu تغییر کند.


### استفاده از مدل

با استفاده از رابط کاربری زیر میتوان از مدل برای شناسایی پهپاد در تصاویر استفاده کرد.

</div>


```
from core import Core  

c = Core()  
  
image_filename = c.current_path + "/DataSets/Drones/testImages/351.jpg"  
image = c.load_image_by_path(image_filename)  
  
drawing_image = c.get_drawing_image(image)  
  
processed_image, scale = c.pre_process_image(image)  
  
c.set_model(c.get_model())  
boxes, scores, labels = c.predict_with_graph_loaded_model(processed_image, scale)  
  
detections = c.draw_boxes_in_image(drawing_image, boxes, scores)  
  
c.visualize(drawing_image)
```

