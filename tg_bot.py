import telebot
import cv2
import numpy as np
import io
import requests
import json
import base64

class PlantAnalysisBot:
    def __init__(self, token, api_url="http://localhost:5000"):
        self.bot = telebot.TeleBot(token)
        self.api_url = api_url
        self.setup_handlers()
    
    def setup_handlers(self):
        @self.bot.message_handler(commands=['start', 'help'])
        def send_welcome(message):
            self.bot.reply_to(message, 
                "🌳 Привет! Я бот для анализа растений от сервиса Green Scan!\n\n"
                "📸 Отправь мне фотографию растений, и я:\n"
                "• Определю породу деревьев/кустов\n"
                "• Обнаружу дефекты и болезни\n"
                "• Покажу результаты с bounding boxes\n\n"
                "Просто отправь фото! 🖼️")
        
        @self.bot.message_handler(content_types=['photo'])
        def handle_photo(message):
            self.process_photo(message)
        
        @self.bot.message_handler(func=lambda message: True)
        def handle_other_messages(message):
            self.bot.reply_to(message, 
                "❌ Я умею работать только с фото следующих форматов: JPG, PNG\n\n"
                "📸 Пожалуйста, отправьте изображение с растениями для анализа!")
    
    def process_photo(self, message):
        try:
            processing_msg = self.bot.reply_to(message, "🔍 Обрабатываю изображение...")
            
            # Скачиваем фото
            file_info = self.bot.get_file(message.photo[-1].file_id)
            downloaded_file = self.bot.download_file(file_info.file_path)
            
            # Отправляем на ваш API
            files = {'file': ('image.jpg', downloaded_file, 'image/jpeg')}
            response = requests.post(f"{self.api_url}/upload", files=files)
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get('no_objects_detected'):
                    self.bot.edit_message_text("❌ На фото не обнаружены деревья или кустарники 🌱", 
                                            message.chat.id, processing_msg.message_id)
                    # Отправляем сообщение о возможности отправить следующее фото
                    self.bot.send_message(message.chat.id, "📸 Можете присылать следующее фото для анализа!")
                    return
                
                # Декодируем обработанное изображение
                image_data = base64.b64decode(result['image'])
                photo = io.BytesIO(image_data)
                photo.name = 'analyzed_image.jpg'
                
                # Формируем отчет
                report = self.format_report(result['table_data'])
                
                # Отправляем результат
                self.bot.send_photo(message.chat.id, photo, caption=report, parse_mode='Markdown')
                self.bot.delete_message(message.chat.id, processing_msg.message_id)
                
                # Отправляем сообщение о возможности отправить следующее фото
                self.bot.send_message(message.chat.id, "✅ Анализ завершен! 📸 Можете присылать следующее фото для анализа!")
                
            else:
                self.bot.edit_message_text("❌ Ошибка при обработке на сервере ⚠️", 
                                        message.chat.id, processing_msg.message_id)
                # Отправляем сообщение о возможности повторить
                self.bot.send_message(message.chat.id, "🔄 Попробуйте отправить фото еще раз!")
                
        except Exception as e:
            self.bot.reply_to(message, f"❌ Ошибка: {str(e)}")
            # Отправляем сообщение о возможности повторить
            self.bot.send_message(message.chat.id, "🔄 Попробуйте отправить фото еще раз!")
    
    def format_report(self, table_data):
        report = "📊 *Результаты анализа:*\n\n"
        
        for item in table_data:
            report += f"🌿 *Объект #{item['id']}*\n"
            report += f"• Тип: {item['plant_type']}\n"
            report += f"• Порода: {item['species']} ({item['species_confidence']}%)\n"
            
            if item.get('species_alt'):
                report += f"• Альтернативная порода: {item['species_alt']['name']} ({item['species_alt_confidence']}%)\n"
            
            report += f"• Состояние: {item['status']} ({item['defects_confidence']}%)\n"
            
            if item.get('defects_alt'):
                report += f"• Альтернативный диагноз: {item['defects_alt']['name']} ({item['defects_alt_confidence']}%)\n"
            
            report += f"• 🔍 Найдено дефектов: {item['defects_count']}\n\n"
        
        return report
    
    def run(self):
        print("🤖 Telegram бот запущен...")
        print("📡 Ожидание сообщений...")
        self.bot.infinity_polling()

if __name__ == "__main__":
    bot = PlantAnalysisBot("8204919684:AAH8Z3X7aq-Kabsdn7p-MDkKWIHAZgzRFUE")
    bot.run()