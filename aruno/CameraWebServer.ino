#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

// 📸 Chọn loại board (AI Thinker)
#define CAMERA_MODEL_AI_THINKER
#include "camera_pins.h"

// 🌐 Wi-Fi cấu hình
const char* ssid = "Gi Cung Duoc";
const char* password = "20032004";

// 🖥️ Server Flask cấu hình
const char* serverIP = "192.168.184.100";
const int serverPort = 5000;

// 🌐 Web server cho camera stream
WebServer server(80);

// 🎤 Cấu hình chân microphone (MAX4466 OUT nối vào GPIO14)
#define MIC_PIN 14
#define SOUND_THRESHOLD 1500  // Ngưỡng âm thanh lớn
#define AUDIO_BUFFER_SIZE 1024 // Kích thước buffer âm thanh

// 📊 Biến toàn cục
uint16_t audioBuffer[AUDIO_BUFFER_SIZE];
bool isRecording = false;

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(false);
  Serial.println();
  
  // 📸 Cấu hình camera AI Thinker
  initCamera();
  
  // 🌐 Kết nối Wi-Fi
  connectWiFi();
  
  // 📡 Đăng ký ESP32 với server Flask
  registerWithServer();
  
  // 🔌 Khởi động server HTTP stream camera
  server.on("/stream", HTTP_GET, handle_jpeg_stream);
  server.begin();
  Serial.println("🌐 HTTP server đã khởi động");
  Serial.println("🔗 Truy cập: http://" + WiFi.localIP().toString() + "/stream");
  
  // 🎤 Cấu hình chân mic
  pinMode(MIC_PIN, INPUT);
  analogSetAttenuation(ADC_11db); // Mở rộng dải ADC
  
  Serial.println("✅ Hệ thống khởi động hoàn tất!");
}

void loop() {
  server.handleClient();
  
  // 🎤 Đọc giá trị microphone
  int micValue = analogRead(MIC_PIN);
  Serial.print("🎤 Âm thanh: ");
  Serial.println(micValue);
  
  // ⚠️ Phát hiện âm thanh lớn
  if (micValue > SOUND_THRESHOLD) {
    Serial.println("⚠️ Âm thanh lớn phát hiện! Có thể là tiếng khóc.");
    
    // 📸 Chụp ảnh
    capture_photo();
    
    // 🎵 Thu âm và gửi lên server
    recordAndSendAudio();
    
    delay(2000);  // Tránh trigger liên tục
  }
  
  delay(100);  // Đọc mic mỗi 100ms
}

void initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size   = FRAMESIZE_QVGA;
  config.jpeg_quality = 12;
  config.fb_count     = 1;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("❌ Camera init failed: 0x%x\n", err);
    return;
  }
  Serial.println("✅ Camera khởi động thành công");
}

void connectWiFi() {
  WiFi.begin(ssid, password);
  Serial.print("⏳ Đang kết nối WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.println("✅ WiFi đã kết nối");
  Serial.print("📡 IP ESP32: ");
  Serial.println(WiFi.localIP());
}

void registerWithServer() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("❌ WiFi chưa kết nối, không thể đăng ký server");
    return;
  }

  HTTPClient http;
  http.begin("http://" + String(serverIP) + ":" + String(serverPort) + "/esp32/register");
  http.addHeader("Content-Type", "application/json");
  
  String payload = "{\"ip\":\"" + WiFi.localIP().toString() + "\"}";
  int httpResponseCode = http.POST(payload);
  
  if (httpResponseCode > 0) {
    Serial.println("✅ Đăng ký thành công với Flask server");
    String response = http.getString();
    Serial.println("📡 Server response: " + response);
  } else {
    Serial.printf("❌ Lỗi đăng ký server: %d\n", httpResponseCode);
  }
  http.end();
}

void handle_jpeg_stream() {
  WiFiClient client = server.client();
  String response = "HTTP/1.1 200 OK\r\n";
  response += "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
  client.print(response);

  while (client.connected()) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("❌ Không lấy được ảnh từ camera");
      continue;
    }

    response = "--frame\r\n";
    response += "Content-Type: image/jpeg\r\n";
    response += "Content-Length: " + String(fb->len) + "\r\n\r\n";
    client.print(response);
    client.write(fb->buf, fb->len);
    client.print("\r\n");
    
    esp_camera_fb_return(fb);
    delay(50);
    
    if (!client.connected()) break;
  }
}

void capture_photo() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("❌ Chụp ảnh thất bại");
    return;
  }
  
  Serial.printf("📸 Đã chụp ảnh, kích thước: %d bytes\n", fb->len);
  
  // TODO: Có thể gửi ảnh lên server nếu cần
  // sendPhotoToServer(fb->buf, fb->len);
  
  esp_camera_fb_return(fb);
}

void recordAndSendAudio() {
  Serial.println("🎵 Bắt đầu thu âm...");
  isRecording = true;
  
  // Thu âm trong 2 giây
  unsigned long startTime = millis();
  int sampleIndex = 0;
  
  while (millis() - startTime < 2000 && sampleIndex < AUDIO_BUFFER_SIZE) {
    audioBuffer[sampleIndex] = analogRead(MIC_PIN);
    sampleIndex++;
    delayMicroseconds(125); // Tần số lấy mẫu 8kHz
  }
  
  isRecording = false;
  Serial.printf("🎵 Hoàn tất thu âm %d samples\n", sampleIndex);
  
  // Chuyển đổi sang bytes để gửi
  size_t audioSize = sampleIndex * 2; // 2 bytes per sample
  uint8_t* audioData = (uint8_t*)audioBuffer;
  
  // Gửi âm thanh lên server
  sendAudioToServer(audioData, audioSize);
}

void sendAudioToServer(uint8_t* audioData, size_t audioSize) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("❌ WiFi chưa kết nối, không thể gửi âm thanh");
    return;
  }

  HTTPClient http;
  http.begin("http://" + String(serverIP) + ":" + String(serverPort) + "/predict");
  http.addHeader("Content-Type", "audio/wav");
  http.addHeader("ESP32-IP", WiFi.localIP().toString());
  http.setTimeout(10000); // 10 seconds timeout
  
  Serial.println("📤 Đang gửi âm thanh lên server...");
  int httpResponseCode = http.POST(audioData, audioSize);
  
  if (httpResponseCode == 200) {
    String response = http.getString();
    Serial.println("📡 Server response: " + response);
    
    // Parse JSON response
    DynamicJsonDocument doc(1024);
    DeserializationError error = deserializeJson(doc, response);
    
    if (!error) {
      bool cryDetected = doc["cry_detected"];
      float confidence = doc["confidence"];
      
      Serial.printf("🔍 Kết quả phân tích: Cry=%s, Confidence=%.2f\n", 
                    cryDetected ? "YES" : "NO", confidence);
      
      if (cryDetected && confidence > 0.7) {
        Serial.println("🚨 TIẾNG KHÓC PHÁT HIỆN! CẢNH BÁO!");
        // Thực hiện hành động khẩn cấp
        triggerAlert();
      }
    } else {
      Serial.println("❌ Lỗi parse JSON response");
    }
  } else {
    Serial.printf("❌ Lỗi gửi âm thanh: HTTP %d\n", httpResponseCode);
  }
  
  http.end();
}

void triggerAlert() {
  // Bạn có thể thêm các hành động cảnh báo ở đây:
  // - Bật đèn LED
  // - Gửi notification qua Telegram/Email
  // - Kích hoạt buzzer
  // - Gửi SMS
  Serial.println("🔔 Kích hoạt cảnh báo tiếng khóc!");
  
  // Ví dụ: Nhấp nháy built-in LED
  for (int i = 0; i < 10; i++) {
    digitalWrite(2, HIGH);
    delay(100);
    digitalWrite(2, LOW);
    delay(100);
  }
}
