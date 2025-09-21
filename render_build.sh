#!/usr/bin/env bash
# render-build.sh
set -e  # Stop on any error

echo "=== بدء عملية البناء على Render ==="

# إنشاء مجلد للبيئات الافتراضية إذا لم يكن موجوداً
mkdir -p .venv

# ترقية pip أولاً
echo "ترقية pip..."
python -m pip install --upgrade pip

# تثبيت setuptools و wheel بشكل منفصل أولاً
echo "تثبيت setuptools و wheel..."
pip install setuptools==65.5.0 wheel==0.41.2

# تثبيت المتطلبات الأساسية
echo "تثبيت المتطلبات الأساسية..."
pip install numpy==1.24.3 Pillow==10.0.1

# ثم تثبيت باقي المتطلبات
echo "تثبيت باقي المتطلبات..."
pip install -r requirements.txt

echo "=== اكتملت عملية البناء بنجاح ==="