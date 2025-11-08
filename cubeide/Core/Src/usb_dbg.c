#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "main.h"
#include "lcd.h"
#include "usb_dbg.h"
#include <usb_host.h>

// This file provides a small __io_putchar implementation so that
// USBH_UsrLog / printf output is visible on the LCD HUD.
// It accumulates characters into a short line buffer and, on '\n' or
// when the buffer fills, copies the line into the shared
// `usb_last_event` string and updates `usb_last_event_time_ms` so the
// main HUD can show the latest host logs.

// Keep buffer modest to avoid heavy SPI usage.
#define DBG_LINE_BUF_SZ 256
static char dbg_line[DBG_LINE_BUF_SZ];
static uint32_t dbg_idx = 0;

// Global circular log buffer to store full lines for HUD display
#define USB_LOG_LINES 32
#define USB_LOG_LINE_SZ 128
char usb_log_lines[USB_LOG_LINES][USB_LOG_LINE_SZ];
volatile uint32_t usb_log_head = 0; // next write index
volatile uint32_t usb_log_count = 0; // number of stored lines (<= USB_LOG_LINES)

/* Runtime guard for whether USB host debug printing is enabled. This should
 * remain 0 during enumeration to avoid heavy printf activity. Set to 1 after
 * enumeration/class selection is complete to enable debug prints.
 *
 * Initialize from the build-time macro `USBH_RUNTIME_LOG_ENABLE_AT_BOOT` if
 * available so a single header change controls boot-time behavior.
 */
#ifdef USBH_RUNTIME_LOG_ENABLE_AT_BOOT
volatile uint8_t USBH_RUNTIME_LOG_ENABLE = USBH_RUNTIME_LOG_ENABLE_AT_BOOT;
#else
volatile uint8_t USBH_RUNTIME_LOG_ENABLE = 1;
#endif

/* Helper to enable runtime logging from user code */
void usbh_enable_runtime_logging(uint8_t enable)
{
  USBH_RUNTIME_LOG_ENABLE = enable ? 1U : 0U;
}

/* UART forwarding hook: implemented in `usb_dbg_uart.c` as a non-blocking
 * DMA/IT-backed enqueuer. Declare it extern here so logging code can call
 * it without providing another definition. */
extern void usb_dbg_uart_send(const char *buf, unsigned int len);

// Preview string and timestamp used by the HUD and other modules.
// Define them here (previously declared extern elsewhere). Placing the
// definition in the logging module keeps the related state co-located
// with the code that updates it.
char usb_last_event[64] = "";
volatile uint32_t usb_last_event_time_ms = 0;

/* Runtime log category mask. Default: disable all categories to avoid any
 * runtime debug output. Use usb_dbg_set_log_mask()/usb_dbg_enable_category()
 * to enable selected categories at runtime if needed. */
volatile uint32_t usb_log_mask = 0;

void usb_dbg_set_log_mask(uint32_t mask)
{
  usb_log_mask = mask;
}

void usb_dbg_enable_category(uint32_t cat, int enable)
{
  if (enable) usb_log_mask |= cat;
  else usb_log_mask &= ~cat;
}

int __io_putchar(int ch)
{
  // Filter carriage return
  if (ch == '\r') return ch;

  if (ch == '\n' || dbg_idx >= (DBG_LINE_BUF_SZ - 1)) {
    // terminate
    dbg_line[dbg_idx] = '\0';

    // store full line into circular buffer
    // ensure null-termination and truncate to USB_LOG_LINE_SZ-1
    dbg_line[USB_LOG_LINE_SZ - 1] = '\0';
    strncpy(usb_log_lines[usb_log_head], dbg_line, USB_LOG_LINE_SZ - 1);
    usb_log_lines[usb_log_head][USB_LOG_LINE_SZ - 1] = '\0';

    // advance head and count
    usb_log_head = (usb_log_head + 1) % USB_LOG_LINES;
    if (usb_log_count < USB_LOG_LINES) usb_log_count++;

  /* Forward the stored line to UART hook if provided. Use the index of
   * the line we just wrote (head - 1).
   */
  int just_idx = (int)usb_log_head - 1;
  if (just_idx < 0) just_idx += USB_LOG_LINES;
  usb_dbg_uart_send(usb_log_lines[just_idx], (unsigned int)strlen(usb_log_lines[just_idx]));

    // also update short preview for compatibility with existing HUD
    const int MAX_PREVIEW = 60; // keep preview a bit longer
    char tmp[MAX_PREVIEW + 1];
    int i;
    for (i = 0; i < MAX_PREVIEW && dbg_line[i] != '\0'; ++i) tmp[i] = dbg_line[i];
    tmp[i] = '\0';
    memset(usb_last_event, 0, sizeof(usb_last_event));
    strncpy(usb_last_event, tmp, sizeof(usb_last_event) - 1);

    // update timestamp so HUD knows there's a new log
    usb_last_event_time_ms = HAL_GetTick();

    // reset index
    dbg_idx = 0;
    return ch;
  }

  dbg_line[dbg_idx++] = (char)ch;
  return ch;
}

// Minimal getchar stub to satisfy weak reference (not used here)
int __io_getchar(void)
{
  return -1;
}

/* Optional printf-style helper used by application debug code.
 * Compiled only when ENABLE_USB_DEBUG_OUTPUT is set to 1 to avoid
 * pulling vsnprintf/printf machinery into release images.
 */
#if ENABLE_USB_DEBUG_OUTPUT
#include <stdarg.h>
void usb_dbg_printf(const char *fmt, ...)
{
  char buf[256];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  /* Forward the formatted line to UART (non-blocking). */
  usb_dbg_uart_send(buf, (unsigned int)strlen(buf));
}
#else
/* When debug output disabled, provide an empty implementation so calls
 * to USB_DBG_PRINTF compile away without pulling formatting code. */
void usb_dbg_printf(const char *fmt, ...)
{
  (void)fmt;
}
#endif
