/* usb_dbg.h
 * Lightweight USB debug helpers and project-wide debug switch.
 *
 * This header centralizes the compile-time control for debug output. Define
 * ENABLE_USB_DEBUG_OUTPUT=1 in your build settings to enable debug printing
 * (which will pull in a small formatting helper). When disabled (=0) the
 * USBH debug macros and USB_DBG_PRINTF expand to no-ops to minimize code
 * size for release builds.
 */
#ifndef __USB_DBG_H
#define __USB_DBG_H

#include <stdint.h>

/* Global switch to enable debug output across the project. Default is 0
 * (disabled). Define ENABLE_USB_DEBUG_OUTPUT=1 in compiler flags to turn
 * debug output on. */
#ifndef ENABLE_USB_DEBUG_OUTPUT
#define ENABLE_USB_DEBUG_OUTPUT 0
#endif

/* Size of circular log buffer and per-line storage size. Keep in sync with
 * the definition in usb_dbg.c */
#define USB_LOG_LINES 32
#define USB_LOG_LINE_SZ 128

/* Circular buffer storing recent full log lines. Lines are null-terminated
 * and truncated to USB_LOG_LINE_SZ-1 when stored. */
extern char usb_log_lines[USB_LOG_LINES][USB_LOG_LINE_SZ];
extern volatile uint32_t usb_log_head; /* next write index */
extern volatile uint32_t usb_log_count; /* number of stored lines (<= USB_LOG_LINES) */

/* Short preview and timestamp used by legacy HUD code */
extern char usb_last_event[64];
extern volatile uint32_t usb_last_event_time_ms;

/* Log categories (bitmask). Use these to enable/disable groups of logs at
 * runtime without changing the compile-time USBH debug level. */
#define USB_LOG_CAT_ENUM  (1U << 0)
#define USB_LOG_CAT_CDC   (1U << 1)
#define USB_LOG_CAT_DATA  (1U << 2)

/* Runtime mask controlling which categories are forwarded to the runtime
 * logger. Default defined in usb_dbg.c. */
extern volatile uint32_t usb_log_mask;

/* Macro to conditionally log by category. Uses the existing USBH_UsrLog
 * implementation (which itself is gated by USBH_DEBUG_LEVEL and
 * USBH_RUNTIME_LOG_ENABLE). */
#define USBH_CAT_LOG(cat, ...) do { if ((usb_log_mask & (cat))) { USBH_UsrLog(__VA_ARGS__); } } while (0)

/* Small stdio retarget helper (kept minimal). */
int __io_putchar(int ch);

/* Optional platform hook: send a raw buffer to UART. This function is the
 * non-blocking forwarder used for actual data passthrough; it must NOT be
 * treated as a debug-only symbol. The USB->USART2 passthrough depends on
 * this remaining defined and available in release builds. */
void usb_dbg_uart_send(const char *buf, unsigned int len);

/* Helper to adjust log mask at runtime */
void usb_dbg_set_log_mask(uint32_t mask);
void usb_dbg_enable_category(uint32_t cat, int enable);

/* Application-level debug printf helper. Controlled by
 * ENABLE_USB_DEBUG_OUTPUT so that release builds do not pull in
 * formatting/printf code. Typical usage in application code:
 *   USB_DBG_PRINTF("a=%d b=%s", a, b);
 */
#if ENABLE_USB_DEBUG_OUTPUT
/* Implemented in Core/Src/usb_dbg.c */
void usb_dbg_printf(const char *fmt, ...);
#define USB_DBG_PRINTF(...) usb_dbg_printf(__VA_ARGS__)
#else
#define USB_DBG_PRINTF(...) do {} while (0)
#endif

#endif /* __USB_DBG_H */
