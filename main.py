from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import io
import logging
import sys
from typing import List

# إعداد logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fashion MNIST Classification API")

# 🔧 إعداد CORS للسماح بالاتصال من المتصفح
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # يمكنك تحديد نطاقات محددة بدلاً من *
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔑 API KEY للحماية
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
API_KEY = "SECRET_KEY_123"

def check_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return api_key

# 🧠 تحميل الموديل المدرب
try:
    model = tf.keras.models.load_model("fashion_mnist_smart_compact.h5")
    logger.info("✅ تم تحميل النموذج بنجاح")
except Exception as e:
    logger.error(f"❌ خطأ في تحميل النموذج: {e}")
    # استمر بدون نموذج للاختبار
    model = None

# أسماء الفئات في Fashion-MNIST
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# 🛠️ دالة المعالجة المحسنة
def preprocess_image(file_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(file_bytes))
        
        if image.mode != 'L':
            image = image.convert('L')
        
        image_array = np.array(image)
        
        if len(image_array.shape) > 2:
            image_array = image_array[:, :, 0]
        
        if np.mean(image_array) > 127:
            image_array = 255 - image_array
        
        image_resized = cv2.resize(image_array, (28, 28), interpolation=cv2.INTER_AREA)
        image_normalized = image_resized.astype("float32") / 255.0
        image_enhanced = np.clip((image_normalized - 0.5) * 1.5 + 0.5, 0, 1)
        image_final = np.expand_dims(image_enhanced, axis=(0, -1))
        
        return image_final
        
    except Exception as e:
        logger.error(f"❌ خطأ في معالجة الصورة: {e}")
        raise HTTPException(status_code=400, detail=f"خطأ في معالجة الصورة: {str(e)}")

# 🛠️ دالة للتحقق من جودة الصورة
def validate_image_quality(image_array: np.ndarray) -> bool:
    if np.max(image_array) - np.min(image_array) < 0.1:
        return False
    if np.std(image_array) < 0.05:
        return False
    return True

@app.get("/")
async def root():
    return {"message": "Fashion MNIST Classification API", "status": "active"}

@app.get("/health")
async def health_check():
    """فحص صحة الخادم"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "classes_count": len(class_names),
        "message": "API is working correctly"
    }

@app.get("/api/classes")
async def get_classes():
    """الحصول على قائمة الفئات المدربة"""
    return {"classes": class_names}

@app.post("/api/predict", dependencies=[Depends(check_api_key)])
async def predict_image(file: UploadFile = File(...)):
    """
    تصنيف صورة من ملابس Fashion-MNIST
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="يجب رفع ملف صورة فقط")
    
    if model is None:
        raise HTTPException(status_code=500, detail="النموذج غير محمل")

    try:
        file_bytes = await file.read()
        
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="الملف فارغ")
        
        image = preprocess_image(file_bytes)
        
        if not validate_image_quality(image[0, :, :, 0]):
            raise HTTPException(status_code=400, detail="الصورة غير واضحة أو لا تحتوي على تباين كافٍ")
        
        prediction = model.predict(image, verbose=0)[0]
        predicted_idx = int(np.argmax(prediction))
        confidence = float(prediction[predicted_idx])
        
        top3_indices = np.argsort(prediction)[-3:][::-1]
        top3_predictions = [
            {
                "class_name": class_names[i],
                "confidence": round(float(prediction[i]) * 100, 2)
            }
            for i in top3_indices
        ]
        
        logger.info(f"✅ تنبؤ ناجح: {class_names[predicted_idx]} ({confidence:.2%})")
        
        return {
            "predicted_class": class_names[predicted_idx],
            "confidence": round(confidence * 100, 2),
            "all_predictions": top3_predictions,
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ خطأ غير متوقع: {e}")
        raise HTTPException(status_code=500, detail=f"خطأ داخلي في الخادم: {str(e)}")

@app.post("/api/predict-batch", dependencies=[Depends(check_api_key)])
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    تصنيف مجموعة من الصور
    """
    results = []
    
    for file in files:
        try:
            if not file.content_type.startswith("image/"):
                results.append({
                    "filename": file.filename,
                    "error": "ليس ملف صورة",
                    "status": "error"
                })
                continue
            
            file_bytes = await file.read()
            image = preprocess_image(file_bytes)
            
            if not validate_image_quality(image[0, :, :, 0]):
                results.append({
                    "filename": file.filename,
                    "error": "صورة غير واضحة",
                    "status": "error"
                })
                continue
            
            prediction = model.predict(image, verbose=0)[0]
            predicted_idx = int(np.argmax(prediction))
            confidence = float(prediction[predicted_idx])
            
            results.append({
                "filename": file.filename,
                "predicted_class": class_names[predicted_idx],
                "confidence": round(confidence * 100, 2),
                "status": "success"
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e),
                "status": "error"
            })
    
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)