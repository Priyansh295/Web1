#include <Servo.h>

Servo myServo;
const int servoPin = 9;

void setup() {
  myServo.attach(servoPin);
  Serial.begin(9600); // Communication with the computer
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    int angle = command.toInt();
    if (angle >= 0 && angle <= 180) {
      myServo.write(angle); // Set the servo to the specified angle
    }
  }
}