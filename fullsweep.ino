#include <SoftWire.h>

// All time values and variables are in milliseconds.
#define MEASUREMENT_TIME    100
// For 16-bit precision there can be 15 samples per second with 66.7 ms for each.
#define ADC_CONVERSION_TIME 80

#define SREG_CLOCK_PIN    2
#define SREG_DATA_PIN     3
#define SREG_STROBE_PIN   4
#define ADC_SDA_PIN       6
#define ADC_SCL_PIN       7
#define BUTTON_PIN        8

#define MCP3425_I2C_ADDR  0x68
// One-shot conversion mode, 16-bit precision, 1 V/V gain.
#define MCP3425_CONFIG            0b00001000
#define MCP3425_START_CONVERSION  0b10001000

unsigned int blinkTimer = 0;

SoftWire myI2C(ADC_SDA_PIN, ADC_SCL_PIN);
byte I2C_TxBuffer[16];
byte I2C_RxBuffer[16];

/********** SHIFT REGISTERS **********/

void setSwitchState(uint16_t state)
{
  // Freeze the outputs
  digitalWrite(SREG_STROBE_PIN, LOW);

  for (unsigned int i = 0; i < 16; i++) {
    // We need to flip the state, because the first serial bit is the last output bit!
    digitalWrite(SREG_DATA_PIN, (state >> (15 - i)) & 1);

    // The maximum frequency is 10 MHz! But we're doing 500 kHz.
    digitalWrite(SREG_CLOCK_PIN, LOW);
    delayMicroseconds(1);
    digitalWrite(SREG_CLOCK_PIN, HIGH);
    delayMicroseconds(1);
  }

  // Allow the ouputs to refresh
  digitalWrite(SREG_STROBE_PIN, HIGH);
}

void clearSwitches()
{
  // Freeze the outputs
  digitalWrite(SREG_STROBE_PIN, LOW);

  for (unsigned int i = 0; i < 16; i++) {
    digitalWrite(SREG_DATA_PIN, LOW);

    // The maximum frequency is 10 MHz! But we're doing 500 kHz.
    digitalWrite(SREG_CLOCK_PIN, LOW);
    delayMicroseconds(1);
    digitalWrite(SREG_CLOCK_PIN, HIGH);
    delayMicroseconds(1);
  }

  // Allow the ouputs to refresh
  digitalWrite(SREG_STROBE_PIN, HIGH);
}

/********** ANALOG TO DIGITAL CONVERTER **********/

/*
  The MCP3425 communication code is partly based on
    https://github.com/ControlEverythingCommunity/MCP3425/tree/master,
    retrieved on 12 March 2025
  and later edited according to the datasheet.
*/

void initADC()
{
  myI2C.beginTransmission(MCP3425_I2C_ADDR);
  myI2C.write(MCP3425_CONFIG);
  myI2C.endTransmission();
}

uint16_t readADC()
{
  // This seems to clear the R/W bit to enable writing.
  myI2C.beginTransmission(MCP3425_I2C_ADDR);
  myI2C.write(MCP3425_START_CONVERSION);
  myI2C.endTransmission();
  
  delay(ADC_CONVERSION_TIME);

  // This seems to set the R/W bit to enable reading.
  myI2C.requestFrom(MCP3425_I2C_ADDR, 3);

  // Wait until data bytes are ready
  while (myI2C.available() != 3);

  byte upperDataByte = myI2C.read();
  byte lowerDataByte = myI2C.read();
  byte configByte = myI2C.read();
  
  // If the RDY bit is set then the conversion is still in progress.
  if (configByte & 0b10000000)
    Serial.println("ADC conversion is still in progress!");

  return ((uint16_t)upperDataByte << 8) | (uint16_t)lowerDataByte;
}

/********** OTHER CODE **********/

void setup()
{
  pinMode(BUTTON_PIN, INPUT);
  pinMode(LED_BUILTIN, OUTPUT);
  pinMode(SREG_CLOCK_PIN, OUTPUT);
  pinMode(SREG_DATA_PIN, OUTPUT);
  pinMode(SREG_STROBE_PIN, OUTPUT);

  digitalWrite(LED_BUILTIN, LOW);
  digitalWrite(SREG_CLOCK_PIN, LOW);
  digitalWrite(SREG_DATA_PIN, LOW);
  digitalWrite(SREG_STROBE_PIN, HIGH);

  clearSwitches();

  Serial.begin(9600);

  myI2C.setTxBuffer(I2C_TxBuffer, sizeof(I2C_TxBuffer));
  myI2C.setRxBuffer(I2C_RxBuffer, sizeof(I2C_RxBuffer));
  myI2C.setDelay_us(5);
  myI2C.setTimeout(1000);
  myI2C.begin();

  initADC();

  delay(1000);
}

void loop()
{
  /*
    If the button is not pressed then blink the LED and wait a little.
    Otherwise start the experiment.
  */
  if (digitalRead(BUTTON_PIN) == LOW)
  {
    if (blinkTimer > 800)
      digitalWrite(LED_BUILTIN, HIGH);
    else if (blinkTimer > 0)
      digitalWrite(LED_BUILTIN, LOW);

    if (blinkTimer > 2 * 800)
      blinkTimer = 0;

    blinkTimer += 10;
    delay(10);

    return;
  }

  digitalWrite(LED_BUILTIN, LOW);

  Serial.println("##### START OF RESULTS #####\n");

  // Start from one, so we could avoid leaving the feedback circuits open.
  // The order of i and j actually doesn't matter because we need to cover
  // all the combinations anyway.
  for (uint16_t i = 1; i < 256; i++) {
    for (uint16_t j = 1; j < 256; j++) {
      uint16_t state = (i << 8) | j;

      setSwitchState(state);
      delay(MEASUREMENT_TIME);

      uint16_t voltage = readADC();

      Serial.print(state, HEX);
      Serial.print(", ");
      Serial.println(voltage, HEX);
    }
  }
  
  Serial.println("\n##### END OF RESULTS #####");

  clearSwitches();
  blinkTimer = 0;
}
