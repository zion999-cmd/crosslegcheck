/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file            : usb_host.c
  * @version         : v1.0_Cube
  * @brief           : This file implements the USB Host
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/

#include "usb_host.h"
#include "usbh_core.h"
#include "usbh_cdc.h"
#include "usb_dbg.h"

/* Shared diagnostics and CDC RX buffer (defined in main.c) */
extern char usb_last_event[64];
extern volatile uint32_t usb_last_event_time_ms;
extern uint8_t CDC_RX_Buffer[512];

/* The CDC RX buffer is allocated in main.c; declare it here so the host
  application can start the first receive when the CDC class becomes active. */
extern uint8_t CDC_RX_Buffer[512];

/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* USER CODE BEGIN PV */
/* Private variables ---------------------------------------------------------*/

/* USER CODE END PV */

/* USER CODE BEGIN PFP */
/* Private function prototypes -----------------------------------------------*/

/* USER CODE END PFP */

/* USB Host core handle declaration */
USBH_HandleTypeDef hUsbHostFS;
ApplicationTypeDef Appli_state = APPLICATION_IDLE;

/*
 * -- Insert your variables declaration here --
 */
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/*
 * user callback declaration
 */
static void USBH_UserProcess(USBH_HandleTypeDef *phost, uint8_t id);

/*
 * -- Insert your external function declaration here --
 */
/* USER CODE BEGIN 1 */

/* USER CODE END 1 */

/**
  * Init USB host library, add supported class and start the library
  * @retval None
  */
void MX_USB_HOST_Init(void)
{
  /* USER CODE BEGIN USB_HOST_Init_PreTreatment */

  /* USER CODE END USB_HOST_Init_PreTreatment */

  /* Init host Library, add supported class and start the library. */
  if (USBH_Init(&hUsbHostFS, USBH_UserProcess, HOST_FS) != USBH_OK)
  {
    Error_Handler();
  }
  if (USBH_RegisterClass(&hUsbHostFS, USBH_CDC_CLASS) != USBH_OK)
  {
    Error_Handler();
  }
  if (USBH_Start(&hUsbHostFS) != USBH_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USB_HOST_Init_PostTreatment */

  /* USER CODE END USB_HOST_Init_PostTreatment */
}

/*
 * Background task
 */
void MX_USB_HOST_Process(void)
{
  /* USB Host Background task */
  USBH_Process(&hUsbHostFS);
}
/*
 * user callback definition
 */
static void USBH_UserProcess  (USBH_HandleTypeDef *phost, uint8_t id)
{
  /* USER CODE BEGIN CALL_BACK_1 */
  switch(id)
  {
  case HOST_USER_SELECT_CONFIGURATION:
    strncpy(usb_last_event, "SELECT_CONFIGURATION", sizeof(usb_last_event)-1);
    usb_last_event_time_ms = HAL_GetTick();
  break;

  case HOST_USER_DISCONNECTION:
  Appli_state = APPLICATION_DISCONNECT;
    strncpy(usb_last_event, "DISCONNECTION", sizeof(usb_last_event)-1);
    usb_last_event_time_ms = HAL_GetTick();
  break;

  case HOST_USER_CLASS_ACTIVE:
  Appli_state = APPLICATION_READY;
    strncpy(usb_last_event, "CLASS_ACTIVE", sizeof(usb_last_event)-1);
    usb_last_event_time_ms = HAL_GetTick();
    /* Some device firmwares only start sending data once the host asserts
     the control line state (DTR). Some firmwares also require the host to
     set the line coding (baud/format) before sending data. First attempt
     to set the line coding to 115200 8N1, then assert control lines. */
   {
    /* Prepare CDC SetLineCoding payload: 115200 (0x0001C200 little-endian),
     * 1 stop bit, no parity, 8 data bits */
    uint8_t linecoding[LINE_CODING_STRUCTURE_SIZE] = {0x00, 0xC2, 0x01, 0x00, 0x00, 0x00, 0x08};
    /* Build CDC_SET_LINE_CODING request */
    hUsbHostFS.Control.setup.b.bmRequestType = USB_H2D | USB_REQ_TYPE_CLASS | USB_REQ_RECIPIENT_INTERFACE;
    hUsbHostFS.Control.setup.b.bRequest = CDC_SET_LINE_CODING;
    hUsbHostFS.Control.setup.b.wValue.w = 0U;
    hUsbHostFS.Control.setup.b.wIndex.w = 0U; /* interface number; 0 is commonly the communication interface */
    hUsbHostFS.Control.setup.b.wLength.w = LINE_CODING_STRUCTURE_SIZE;
    {
      USBH_StatusTypeDef lc_status = USBH_CtlReq(&hUsbHostFS, linecoding, LINE_CODING_STRUCTURE_SIZE);
      USBH_CAT_LOG(USB_LOG_CAT_CDC, "CDC: SetLineCoding(115200,8N1) status=%d", (int)lc_status);
      if (lc_status != USBH_OK) {
        USBH_CAT_LOG(USB_LOG_CAT_CDC, "CDC: SetLineCoding failed (status=%d)", (int)lc_status);
      }
    }
   }
  {
   /* Build a class-specific, host-to-device, interface recipient setup */
    hUsbHostFS.Control.setup.b.bmRequestType = USB_H2D | USB_REQ_TYPE_CLASS | USB_REQ_RECIPIENT_INTERFACE;
    hUsbHostFS.Control.setup.b.bRequest = CDC_SET_CONTROL_LINE_STATE;
    hUsbHostFS.Control.setup.b.wValue.w = (uint16_t)(CDC_ACTIVATE_SIGNAL_DTR | CDC_ACTIVATE_CARRIER_SIGNAL_RTS);
    hUsbHostFS.Control.setup.b.wIndex.w = 0U; /* interface number (0 assumed) */
    hUsbHostFS.Control.setup.b.wLength.w = 0U;

    {
    /* Try the control request and retry a few times if it returns BUSY so
     * the device has more chances to accept it. Limit retries to avoid
     * blocking the system for too long. */
  const int max_attempts = 10;
    int attempt = 0;
    USBH_StatusTypeDef req_status = USBH_BUSY;
    for (attempt = 0; attempt < max_attempts; ++attempt) {
  req_status = USBH_CtlReq(&hUsbHostFS, NULL, 0U);
  USBH_CAT_LOG(USB_LOG_CAT_CDC, "CDC: SetControlLineState attempt %d status=%d", attempt + 1, (int)req_status);
      if (req_status == USBH_OK) break;
      HAL_Delay(20);
    }
    if (req_status != USBH_OK) {
      USBH_CAT_LOG(USB_LOG_CAT_CDC, "CDC: SetControlLineState failed after %d attempts (status=%d)", max_attempts, (int)req_status);
    }

    /* small delay to allow device to react */
    HAL_Delay(30);

    /* Start the first CDC receive so that data from the device will be
       actively pushed into our buffer. Subsequent receives are re-queued
       from the CDC receive callback. Log the receive call return value so
       we can see from the UART whether the host successfully queued the
       transfer. */
    {
  USBH_StatusTypeDef recv_status = USBH_CDC_Receive(&hUsbHostFS, CDC_RX_Buffer, sizeof(CDC_RX_Buffer));
  USBH_CAT_LOG(USB_LOG_CAT_CDC, "CDC: First receive req status=%d", (int)recv_status);
    }
   }
   }
  break;

  case HOST_USER_CONNECTION:
  Appli_state = APPLICATION_START;
    strncpy(usb_last_event, "CONNECTION", sizeof(usb_last_event)-1);
    usb_last_event_time_ms = HAL_GetTick();
  break;

  default:
  break;
  }
  /* USER CODE END CALL_BACK_1 */
}

/**
  * @}
  */

/**
  * @}
  */

