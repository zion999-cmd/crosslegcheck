/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : Target/usbh_conf.h
  * @version        : v1.0_Cube
  * @brief          : Header for usbh_conf.c file.
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

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __USBH_CONF__H__
#define __USBH_CONF__H__

#ifdef __cplusplus
 extern "C" {
#endif
/* Includes ------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "main.h"

#include "stm32h7xx.h"
#include "stm32h7xx_hal.h"

/* USER CODE BEGIN INCLUDE */

/* USER CODE END INCLUDE */

/** @addtogroup STM32_USB_HOST_LIBRARY
  * @{
  */

/** @defgroup USBH_CONF
  * @brief usb host low level driver configuration file
  * @{
  */

/** @defgroup USBH_CONF_Exported_Variables USBH_CONF_Exported_Variables
  * @brief Public variables.
  * @{
  */

/**
  * @}
  */

/** @defgroup USBH_CONF_Exported_Defines USBH_CONF_Exported_Defines
  * @brief Defines for configuration of the Usb host.
  * @{
  */

/*----------   -----------*/
#define USBH_MAX_NUM_ENDPOINTS      2U

/*----------   -----------*/
#define USBH_MAX_NUM_INTERFACES      2U

/*----------   -----------*/
#define USBH_MAX_NUM_CONFIGURATION      1U

/*----------   -----------*/
#define USBH_KEEP_CFG_DESCRIPTOR      1U

/*----------   -----------*/
#define USBH_MAX_NUM_SUPPORTED_CLASS      1U

/*----------   -----------*/
#define USBH_MAX_SIZE_CONFIGURATION      256U

/*----------   -----------*/
#define USBH_MAX_DATA_BUFFER      512U

/*----------   -----------*/
/* Control USB Host debug level. Tie the USBH debug level to the global
 * ENABLE_USB_DEBUG_OUTPUT compile-time switch when possible. If
 * ENABLE_USB_DEBUG_OUTPUT is defined (e.g. -DENABLE_USB_DEBUG_OUTPUT=1)
 * the USBH debug level will default to verbose (3). Otherwise it will
 * default to 0 and produce no printf output in the library. The macro
 * can still be overridden by defining USBH_DEBUG_LEVEL explicitly. */
#ifndef USBH_DEBUG_LEVEL
#if defined(ENABLE_USB_DEBUG_OUTPUT) && (ENABLE_USB_DEBUG_OUTPUT != 0)
#define USBH_DEBUG_LEVEL 3U
#else
#define USBH_DEBUG_LEVEL 0U
#endif
#endif

/* Runtime switch: when 0, debug macros become no-ops; set to 1 after
 * enumeration/initialization is complete to enable full logging. The
 * variable is defined in `usb_dbg.c`.
 */
extern volatile uint8_t USBH_RUNTIME_LOG_ENABLE;

/* Optional: skip slow string-descriptor reads during enumeration.
 * Some USB devices respond very slowly to string descriptor requests
 * (manufacturer/product/serial) which can hang enumeration for many
 * seconds. Enabling this will omit those reads and complete
 * enumeration faster. Set to 1 to enable; 0 to keep default behavior.
 */
#define USBH_SKIP_STRING_DESC  1U

/*----------   -----------*/
#define USBH_USE_OS      0U

/* Optional: enable endpoint-based fallback to force-bind CDC class
 * when the interface class code does not match but the endpoint layout
 * looks like a CDC device (one interrupt IN + one bulk IN + one bulk OUT).
 * Set to 1 to enable this experimental hack for devices that report
 * a vendor-specific class but expose CDC-like endpoints.
 */
#define USBH_FORCE_CDC_BY_ENDPOINT 1U

/****************************************/
/* #define for FS and HS identification */
#define HOST_HS 		0
#define HOST_FS 		1

#if (USBH_USE_OS == 1)
  #include "cmsis_os.h"
  #define USBH_PROCESS_PRIO          osPriorityNormal
  #define USBH_PROCESS_STACK_SIZE    ((uint16_t)0)
#endif /* (USBH_USE_OS == 1) */

/**
  * @}
  */

/** @defgroup USBH_CONF_Exported_Macros USBH_CONF_Exported_Macros
  * @brief Aliases.
  * @{
  */

/* Memory management macros */

/** Alias for memory allocation. */
#define USBH_malloc         malloc

/** Alias for memory release. */
#define USBH_free           free

/** Alias for memory set. */
#define USBH_memset         memset

/** Alias for memory copy. */
#define USBH_memcpy         memcpy

/* DEBUG macros */

#if (USBH_DEBUG_LEVEL > 0U)
#define  USBH_UsrLog(...)   do { \
                            if (USBH_RUNTIME_LOG_ENABLE) { \
                              printf(__VA_ARGS__); \
                              printf("\n"); \
                            } \
} while (0)
#else
#define USBH_UsrLog(...) do {} while (0)
#endif

#if (USBH_DEBUG_LEVEL > 1U)

#define  USBH_ErrLog(...) do { \
                            if (USBH_RUNTIME_LOG_ENABLE) { \
                              printf("ERROR: "); \
                              printf(__VA_ARGS__); \
                              printf("\n"); \
                            } \
} while (0)
#else
#define USBH_ErrLog(...) do {} while (0)
#endif

#if (USBH_DEBUG_LEVEL > 2U)
#define  USBH_DbgLog(...)   do { \
                            if (USBH_RUNTIME_LOG_ENABLE) { \
                              printf("DEBUG : "); \
                              printf(__VA_ARGS__); \
                              printf("\n"); \
                            } \
} while (0)
#else
#define USBH_DbgLog(...) do {} while (0)
#endif

/**
  * @}
  */

/** @defgroup USBH_CONF_Exported_Types USBH_CONF_Exported_Types
  * @brief Types.
  * @{
  */

/**
  * @}
  */

/** @defgroup USBH_CONF_Exported_FunctionsPrototype USBH_CONF_Exported_FunctionsPrototype
  * @brief Declaration of public functions for Usb host.
  * @{
  */

/* Exported functions -------------------------------------------------------*/

/**
  * @}
  */

/**
  * @}
  */

/**
  * @}
  */

#ifdef __cplusplus
}
#endif
#endif /* __USBH_CONF__H__ */
