const int ledPin = 7;  // Pin to which the LED is connected

void setup() {
  pinMode(ledPin, OUTPUT);  // Set the LED pin as an output
  digitalWrite(ledPin, LOW);  // Ensure the LED is off at startup
  Serial.begin(9600);  // Start serial communication at 9600 baud
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');  // Read the command from serial input
    command.trim();  // Remove any leading or trailing whitespace
    if (command == "TURN ON") {
      digitalWrite(ledPin, HIGH);  // Turn on the LED
    } else if (command == "TURN OFF") {
      digitalWrite(ledPin, LOW);  // Turn off the LED
    }
  }
}