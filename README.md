# خلاصه‌سازی دیالوگ با T5 و تکنیک‌های تنظیم دقیق (LoRA و Full Fine-Tuning)

در این پروژه یک سیستم خلاصه‌سازی خودکار برای دیالوگ‌های دو نفره پیاده‌سازی شده است که شامل **پیش‌پردازش دیتاست، تنظیم دقیق مدل T5-base** به دو روش کامل و به‌صرفه (LoRA)، **ارزیابی کمی با معیار ROUGE**، و **تولید گزارش کیفی از نتایج** می‌باشد.  
هدف اصلی پروژه، مقایسه‌ی عملکرد مدل پایه، مدل فاین‌تیون کامل و مدل تنظیم‌شده با LoRA در یک مسئله‌ی خلاصه‌سازی متنی است.

---
## 🏷️ Badges
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch" />
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface" />
  <img src="https://img.shields.io/badge/PEFT-LoRA-success" />
  <img src="https://img.shields.io/badge/Status-Active-brightgreen" />
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" />
</p>

---
## 📦 Dataset

### 🔹 دیتاست مورد استفاده – DialogSum
دیتاست **DialogSum** شامل دیالوگ‌های کوتاه دو نفره به همراه خلاصه‌های دست‌نویس انسانی است. این دیتاست به‌طور مستقیم از Hugging Face Datasets بارگذاری می‌شود.

- **تعداد نمونه‌های آموزش:** ۱۲,۴۶۰  
- **تعداد نمونه‌های اعتبارسنجی:** ۵۰۰  
- **تعداد نمونه‌های تست:** ۱,۵۰۰  

🔗  
https://huggingface.co/datasets/knkarthick/dialogsum

---
## مراحل انجام پروژه

### 1️⃣ پیش‌پردازش و توکنایز کردن دیتاست
در این مرحله، دیالوگ‌ها با استفاده از قالب پرامپت زیر به ورودی مدل تبدیل شدند:
