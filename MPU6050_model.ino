//TFlite_MPU6050_v1.ino
#include <TensorFlowLite_ESP32.h>
#include "model.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"

#include "tensorflow/lite/schema/schema_generated.h"

#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>


#define THRESHOLD 20
#define READINGS_PER_SAMPLE 40

Adafruit_MPU6050 mpu;

float ax;
float ay;
float az;
float baseAx;
float baseAy;
float baseAz;


namespace{
const tflite::Model*  tflModel; 
tflite::ErrorReporter*  tflErrorReporter; 
constexpr int tensorArenaSize = 102 * 1024; 
uint8_t tensorArena[tensorArenaSize];
TfLiteTensor* tflInputTensor; 
TfLiteTensor* tflOutputTensor; 
tflite::MicroInterpreter* tflInterpreter; 
}

#define NUM_GESTURES  1

void setup() {

  Serial.begin(115200);
  while (!Serial)
    delay(10); 

  Serial.println("Adafruit MPU6050 test!");
  // Try to initialize!
  
  if (!mpu.begin()) {
    Serial.println("Failed to find MPU6050 chip");
    while (1) {
      delay(10);
    }
  }
  
  Serial.println("MPU6050 Found!");
  // Set Accelaration Range
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);

  calibrate_sensor();
  Serial.println("");
  
  // put your setup code here, to run once:
  static tflite::MicroErrorReporter micro_error_reporter; 
  tflErrorReporter = &micro_error_reporter;

   tflModel = tflite::GetModel(g_model);
   if (tflModel->version() != TFLITE_SCHEMA_VERSION) {

    TF_LITE_REPORT_ERROR(tflErrorReporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         tflModel->version(), TFLITE_SCHEMA_VERSION);

    return;
  }

  static tflite::MicroMutableOpResolver<2> micro_mutable_op_resolver(tflErrorReporter);  

  if (micro_mutable_op_resolver.AddFullyConnected() != kTfLiteOk) {
    return;
  }
  if (micro_mutable_op_resolver.AddSoftmax() != kTfLiteOk) {
    return;
  }
  //micro_mutable_op_resolver.AddSoftmax();

  /*
  micro_mutable_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_FULLY_CONNECTED,
      tflite::ops::micro::Register_FULLY_CONNECTED());
      //tflite::Register_FULLY_CONNECTED());
  */
      
  static tflite::MicroInterpreter static_interpreter(tflModel, micro_mutable_op_resolver, tensorArena, tensorArenaSize, tflErrorReporter);
  tflInterpreter = &static_interpreter;
  
  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = tflInterpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(tflErrorReporter, "AllocateTensors() failed");
    return;
  }

  Serial.print("setup complete");
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);
  
}

void loop() {
  // put your main code here, to run repeatedly:
  detectMovement();
}

void run_inference(){
  sensors_event_t a, g, temp;
  for(int i =0; i< READINGS_PER_SAMPLE; i++){
  mpu.getEvent(&a, &g, &temp);
  ax = a.acceleration.x - baseAx;
  ay = a.acceleration.y - baseAy;
  az = a.acceleration.z - baseAz;
  tflInputTensor->data.f[i * 3 + 0] = (ax + 8.0) / 16.0;
  tflInputTensor->data.f[i * 3 + 1] = (ay + 8.0) / 16.0;
  tflInputTensor->data.f[i * 3 + 2] = (az + 8.0) / 16.0;
  delay(10);
  }

  TfLiteStatus invokeStatus = tflInterpreter->Invoke();
  float out = tflOutputTensor->data.f[1];
  if(out >= 0.80){
    Serial.println("Shoot");
  }
  else{
    Serial.println("Unknown");
  }
  
}
void  detectMovement() {
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);
  if( abs(a.acceleration.x - baseAx) +abs(a.acceleration.y - baseAy) + abs(a.acceleration.z - baseAz) > THRESHOLD){
    run_inference();
  }
  else{
    delay(5);
  }
}
void calibrate_sensor() {
  float totX, totY, totZ;
  sensors_event_t a, g, temp;
  
  for (int i = 0; i < 10; i++) {
    mpu.getEvent(&a, &g, &temp);
    totX = totX + a.acceleration.x;
    totY = totY + a.acceleration.y;
    totZ = totZ + a.acceleration.z;
  }
  baseAx = totX / 10;
  baseAy = totY / 10;
  baseAz = totZ / 10;
}
