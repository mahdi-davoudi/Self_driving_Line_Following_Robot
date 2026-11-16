#include <Servo.h>
#include <NewPing.h>
#include <TM1637Display.h>

// ================= PINS =================
#define CLK 12
#define DIO 13

#define TRIG_PIN A4
#define ECHO_PIN A3
#define MAX_DISTANCE 250

#define ENA 3   // left motor PWM  (remove the ENA jumper)
#define IN1 8
#define IN2 4
#define ENB 5   // right motor PWM (remove the ENB jumper)
#define IN3 7
#define IN4 6

#define HALL_PIN 2   // interrupt pin

TM1637Display display(CLK, DIO);
Servo myservo;
NewPing sonar(TRIG_PIN, ECHO_PIN, MAX_DISTANCE);

// ================= HALL SENSOR =================
#define PULSE_DEBOUNCE_US 500UL   // pulses closer than this are ignored (noise)
#define MOVE_TIMEOUT_MS   5000UL  // stop moving if no pulses arrive (stuck wheel / dead sensor)

// ================= MANEUVER PARAMETERS =================
// NO values here: Python sends all of them at startup ("CFG:<NAME>=<value>"),
// then "CFGEND". Until CFGEND arrives, maneuvers are disabled (cfgReady = false).
//   *_P = Hall pulses to drive, *_A = angle passed to steer(), CW_WAIT_MS = crosswalk stop time
int AV_BACK_P, AV_LEFT_P, AV_MID_P, AV_RIGHT_P, AV_FINAL_P;   // obstacle avoidance pulses
int AV_A_STOP, AV_A_LEFT, AV_A_MID, AV_A_RIGHT, AV_A_FINAL;   // obstacle avoidance angles
int CW_WAIT_MS, CW_FWD_P, CW_A;                               // crosswalk
int FR_P, FR_A;                                               // straight
int RR_1_P, RR_A1, RR_4_P, RR_A2;                             // turn right
int LL_1_P, LL_2_P, LL_A2, LL_3_P, LL_A3;                     // turn left
bool cfgReady = false;

#define DIR_FWD   1
#define DIR_BACK -1

// ================= LIMITS =================
#define MAX_SPEED     210  // absolute PWM safety limit (0..255)

#define SERVO_CENTER  90
#define SERVO_MIN     45
#define SERVO_MAX     135

// Maneuver angles (*_A) were originally tuned for a center of 76.
// steer() shifts them by this offset so they stay relative to the new center (90).
#define SERVO_OFFSET  (SERVO_CENTER - 76)

int current_speed = 200;
int current_angle = SERVO_CENTER;
int pwm_speed = 200;

#define SERIAL_BUFFER_SIZE 32
char serial_buffer[SERIAL_BUFFER_SIZE];
uint8_t buffer_index = 0;

bool avoiding = false;

#define HEARTBEAT_TIMEOUT_MS 500UL

unsigned long lastHeartbeatMs = 0;
bool heartbeatReceived = false;
bool watchdogTripped = true;

// ================= HALL PULSE COUNTER =================
volatile unsigned long pulseCount = 0;
volatile unsigned long lastPulseUs = 0;

void onPulse() {
  unsigned long now = micros();
  if (now - lastPulseUs < PULSE_DEBOUNCE_US) return;  // debounce
  lastPulseUs = now;
  pulseCount++;
}

unsigned long getPulses() {
  noInterrupts();
  unsigned long c = pulseCount;
  interrupts();
  return c;
}

// ================= SERVO =================
// Absolute servo angle, always clamped to [SERVO_MIN, SERVO_MAX]
void servoWrite(int angle) {
  angle = constrain(angle, SERVO_MIN, SERVO_MAX);
  myservo.write(angle);
}

// Maneuver angle (old 76-centered value) -> shifted to the new center, then clamped
void steer(int oldAngle) {
  servoWrite(oldAngle + SERVO_OFFSET);
}

// ================= MOTORS =================
void driveChannel(int in_a, int in_b, int en_pin, int speed) {

  speed = constrain(speed, -MAX_SPEED, MAX_SPEED);

  if (speed > 0) {
    digitalWrite(in_a, HIGH);
    digitalWrite(in_b, LOW);
    analogWrite(en_pin, speed);

  } else if (speed < 0) {
    digitalWrite(in_a, LOW);
    digitalWrite(in_b, HIGH);
    analogWrite(en_pin, -speed);

  } else {
    digitalWrite(in_a, LOW);
    digitalWrite(in_b, LOW);
    analogWrite(en_pin, 0);
  }
}


void motor(int L, int R) {
  if (watchdogTripped && (L != 0 || R != 0)) {
    L = 0;
    R = 0;
  }

  driveChannel(IN1, IN2, ENA, L);
  driveChannel(IN3, IN4, ENB, R);
}




void handleHeartbeat() {
  lastHeartbeatMs = millis();

  if (!heartbeatReceived) {
    heartbeatReceived = true;
    watchdogTripped = false;
    Serial.println("WATCHDOG_ARMED");
  }
}

void checkWatchdog() {
  if (!heartbeatReceived || watchdogTripped) {
    return;
  }

  if (millis() - lastHeartbeatMs >= HEARTBEAT_TIMEOUT_MS) {
    watchdogTripped = true;
    motor(0, 0);
    Serial.println("WATCHDOG_STOP");
  }
}

void serviceHeartbeatSerial() {
  while (Serial.available() > 0) {
    char c = Serial.read();

    if (c == '\r' || c == '\n') {
      if (buffer_index > 0) {
        serial_buffer[buffer_index] = '\0';

        if (strcmp(serial_buffer, "HB") == 0) {
          handleHeartbeat();
        }

        buffer_index = 0;
      }
    } else if (buffer_index < SERIAL_BUFFER_SIZE - 1) {
      serial_buffer[buffer_index++] = c;
    } else {
      buffer_index = 0;
    }
  }
}

bool safeDelay(unsigned long durationMs) {
  unsigned long startMs = millis();

  while (millis() - startMs < durationMs) {
    serviceHeartbeatSerial();
    checkWatchdog();

    if (watchdogTripped) {
      motor(0, 0);
      return false;
    }
  }

  return true;
}


void movePulses(int dir, unsigned long pulses, bool stopAtEnd) {
  if (watchdogTripped) {
    motor(0, 0);
    return;
  }

  unsigned long start = getPulses();
  unsigned long t0 = millis();

  motor(dir * pwm_speed, dir * pwm_speed);

  while ((getPulses() - start) < pulses) {
    serviceHeartbeatSerial();
    checkWatchdog();

    if (watchdogTripped) {
      motor(0, 0);
      return;
    }

    if (millis() - t0 > MOVE_TIMEOUT_MS) {
      motor(0, 0);
      Serial.println("PULSE_TIMEOUT");
      return;
    }
  }

  if (stopAtEnd) {
    motor(0, 0);
  }
}

// ================= PARAMETER UPLOAD =================
// Set one maneuver parameter by name. Names are stored in flash (PSTR) to save RAM.
bool setParam(const char *key, long val) {
  val = constrain(val, 0L, 30000L);
#define SETP(n) if (strcmp_P(key, PSTR(#n)) == 0) { n = (int)val; return true; }
  SETP(AV_BACK_P) SETP(AV_LEFT_P) SETP(AV_MID_P) SETP(AV_RIGHT_P) SETP(AV_FINAL_P)
  SETP(AV_A_STOP) SETP(AV_A_LEFT) SETP(AV_A_MID) SETP(AV_A_RIGHT) SETP(AV_A_FINAL)
  SETP(CW_WAIT_MS) SETP(CW_FWD_P) SETP(CW_A)
  SETP(FR_P) SETP(FR_A)
  SETP(RR_1_P) SETP(RR_A1) SETP(RR_4_P) SETP(RR_A2)
  SETP(LL_1_P) SETP(LL_2_P) SETP(LL_A2) SETP(LL_3_P) SETP(LL_A3)
#undef SETP
  return false;
}

// ================= SETUP =================
void setup() {
  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(ENB, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  pinMode(HALL_PIN, INPUT);
  attachInterrupt(digitalPinToInterrupt(HALL_PIN), onPulse, FALLING);

  Serial.begin(115200);
  display.setBrightness(7);
  myservo.attach(9);
  servoWrite(current_angle);
  motor(0, 0);

  // Moderate PWM at start; Python can change it with SPD<value>.
  pwm_speed = 200;
  current_speed = pwm_speed;

  delay(1000);
}

// ================= LOOP =================
void loop() {
  int d = sonar.ping_cm();
  Serial.println(d);

  // ---- Obstacle avoidance (only after Python has uploaded the parameters) ----
  if (cfgReady && !avoiding && d > 0 && d <= 40) {
    Serial.println(d);
    avoiding = true;

    steer(AV_A_STOP);
    motor(0, 0);
    safeDelay(1000);                                   // stand still
    movePulses(DIR_BACK, AV_BACK_P, false);

    // left
    steer(AV_A_LEFT);
    movePulses(DIR_FWD, AV_LEFT_P, true);
    safeDelay(1000);

    // middle
    steer(AV_A_MID);
    movePulses(DIR_FWD, AV_MID_P, true);
    safeDelay(1000);

    // right, then final correction
    steer(AV_A_RIGHT);
    movePulses(DIR_FWD, AV_RIGHT_P, false);
    steer(AV_A_FINAL);
    movePulses(DIR_FWD, AV_FINAL_P, false);

    servoWrite(SERVO_CENTER);
    motor(0, 0);
    safeDelay(500);

    avoiding = false;
  }

  // ---- Serial command reader ----
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\r' || c == '\n') {
      if (buffer_index > 0) {
        serial_buffer[buffer_index] = '\0';
        processCommand();
        buffer_index = 0;
      }
    } else if (buffer_index < SERIAL_BUFFER_SIZE - 1) {
      serial_buffer[buffer_index++] = c;
    } else {
      buffer_index = 0;
    }
  }
  checkWatchdog();
}

// ================= COMMANDS =================
void processCommand() {

  if (strcmp(serial_buffer, "HB") == 0) {
    handleHeartbeat();
    return;
  }
  // ---- End of parameter upload ----
  if (strcmp(serial_buffer, "CFGEND") == 0) {
    cfgReady = true;
    Serial.println("CFG_OK");
    return;
  }

  // ---- Maneuvers need the parameters from Python ----
  if (!cfgReady && (strcmp(serial_buffer, "C") == 0 || strcmp(serial_buffer, "Fr") == 0 ||
                    strcmp(serial_buffer, "RR") == 0 || strcmp(serial_buffer, "LL") == 0)) {
    Serial.println("CFG_MISSING");
    Serial.println("DONE");   // so Python does not wait for the timeout
    return;
  }

  // ---- Maneuver parameter: CFG:<NAME>=<value> ----
  if (strncmp(serial_buffer, "CFG:", 4) == 0) {
    char *eq = strchr(serial_buffer + 4, '=');
    if (eq == nullptr) { Serial.println("CFG_ERR"); return; }
    *eq = '\0';
    long v = strtol(eq + 1, nullptr, 10);
    Serial.println(setParam(serial_buffer + 4, v) ? "CFG_OK" : "CFG_ERR");
    return;
  }

  // ---- Forward ----
  if (strcmp(serial_buffer, "F") == 0) {
    display.showNumberDec(4);
    motor(pwm_speed, pwm_speed);
    return;
  }

  // ---- Stop sign ----
  if (strcmp(serial_buffer, "SS") == 0) {
    display.showNumberDec(5);
    digitalWrite(A0, 1);
    digitalWrite(A1, 1);
    motor(0, 0);
    servoWrite(current_angle);
    safeDelay(1000);                         // stand still at the STOP sign
    Serial.println("DONE");
    return;
  }

  // ---- Crosswalk ----
  if (strcmp(serial_buffer, "C") == 0) {
    display.showNumberDec(1111);

    motor(0, 0);
    steer(CW_A);
    safeDelay(CW_WAIT_MS);                   // stop at the crosswalk
    servoWrite(SERVO_CENTER);
    safeDelay(300);                          // let the servo reach the center

    movePulses(DIR_FWD, CW_FWD_P, true);
    motor(0, 0);
    safeDelay(2000);

    Serial.println("DONE");
    return;
  }

  // ---- Straight ----
  if (strcmp(serial_buffer, "Fr") == 0) {
    motor(0, 0);
    safeDelay(300);
    steer(FR_A);
    movePulses(DIR_FWD, FR_P, true);
    safeDelay(200);
    Serial.println("DONE");
    return;
  }

  // ---- Turn right ----
  if (strcmp(serial_buffer, "RR") == 0) {
    display.showNumberDec(2);
    digitalWrite(A0, 1);

    steer(RR_A1);
    movePulses(DIR_FWD, RR_1_P, false);
    steer(RR_A2);
    movePulses(DIR_FWD, RR_4_P, false);
    servoWrite(SERVO_CENTER);
    motor(0, 0);
    safeDelay(10);
    Serial.println("DONE");
    return;
  }

  // ---- Turn left ----
  if (strcmp(serial_buffer, "LL") == 0) {
    display.showNumberDec(3);
    digitalWrite(A1, 1);

    movePulses(DIR_FWD, LL_1_P, false);   // keeps the current steering angle
    steer(LL_A2);
    movePulses(DIR_FWD, LL_2_P, false);
    steer(LL_A3);
    movePulses(DIR_FWD, LL_3_P, false);   // motors stay on, Python sends 'F' next
    Serial.println("DONE");
    return;
  }

  if (strcmp(serial_buffer, "ST") == 0) {
    motor(0, 0);
    return;
  }

  // ---- Park ----
  if (strcmp(serial_buffer, "EF") == 0) {
    motor(0, 0);
    Serial.println("DONE");
    return;
  }

  if (strcmp(serial_buffer, "B") == 0) {
    motor(-pwm_speed, -pwm_speed);
    return;
  }

  if (strcmp(serial_buffer, "STOP") == 0) {
    motor(0, 0);
    return;
  }

  // Speed command: SPD
  if (strncmp(serial_buffer, "SPD", 3) == 0) {

      char *endptr = nullptr;

      long requested_speed = strtol(
          serial_buffer + 3,
          &endptr,
          10
      );

      if (endptr == serial_buffer + 3 || *endptr != '\0') {
          return;
      }

      pwm_speed = (int)constrain(
          requested_speed,
          0L,
          (long)MAX_SPEED
      );

      current_speed = pwm_speed;

      return;
  }

  // ---- Steering command: S<angle> (absolute, 45..135) ----
  if (serial_buffer[0] == 'S') {
    int new_angle = atoi(serial_buffer + 1);
    current_angle = constrain(new_angle, SERVO_MIN, SERVO_MAX);
    servoWrite(current_angle);
    return;
  }
}