/* USER CODE BEGIN USB_DBG_UART */
/* Non-blocking UART forwarder with ring buffer and DMA/IT fallback.
 * - Enqueue log lines into a TX ring buffer.
 * - If huart2->hdmatx is configured, use HAL_UART_Transmit_DMA to send
 *   contiguous chunks; otherwise fall back to HAL_UART_Transmit_IT.
 * - Continue sending remaining data in the HAL UART Tx complete callback.
 *
 * This minimizes blocking in the USB/log hot paths while keeping the
 * implementation compatible when DMA is not configured.
 */

#include "main.h"
#include "usb_dbg.h"
#include <string.h>
#include <stdint.h>

extern UART_HandleTypeDef huart2; /* declared in main.c */

/* Ring buffer size - keep moderately large to buffer bursts */
#define USB_DBG_UART_RING_SZ 4096U
static uint8_t usb_dbg_uart_ring[USB_DBG_UART_RING_SZ];
static volatile uint32_t uart_ring_head = 0; /* next write index */
static volatile uint32_t uart_ring_tail = 0; /* next read index */

/* Transmission state */
static volatile uint8_t uart_tx_busy = 0; /* 0 == idle, 1 == sending */
static volatile uint32_t uart_last_sent = 0; /* last sent length (for tail advance) */

/* Internal helpers */
static uint32_t usb_dbg_uart_ring_used(void) {
  uint32_t h = uart_ring_head;
  uint32_t t = uart_ring_tail;
  return (h >= t) ? (h - t) : (USB_DBG_UART_RING_SZ - t + h);
}

static uint32_t usb_dbg_uart_ring_free(void) {
  return USB_DBG_UART_RING_SZ - usb_dbg_uart_ring_used() - 1U; /* keep one byte free */
}

/* Start sending next contiguous chunk from tail. Called with interrupts enabled.
 * Returns 0 if started a transfer, non-zero if nothing to send or busy.
 */
static int usb_dbg_uart_start_tx_if_idle(void) {
  if (uart_tx_busy) return -1;

  uint32_t used = usb_dbg_uart_ring_used();
  if (used == 0) return -2; /* nothing to send */

  /* contiguous bytes from tail to end of buffer */
  uint32_t t = uart_ring_tail;
  uint32_t cont = (t + used <= USB_DBG_UART_RING_SZ) ? used : (USB_DBG_UART_RING_SZ - t);

  uart_tx_busy = 1;
  uart_last_sent = cont;

  /* Prefer DMA when available */
  if (huart2.hdmatx != NULL) {
    if (HAL_UART_Transmit_DMA(&huart2, &usb_dbg_uart_ring[t], (uint16_t)cont) != HAL_OK) {
      /* fallback to IT on error */
      (void)HAL_UART_Transmit_IT(&huart2, &usb_dbg_uart_ring[t], (uint16_t)cont);
    }
  } else {
    (void)HAL_UART_Transmit_IT(&huart2, &usb_dbg_uart_ring[t], (uint16_t)cont);
  }

  return 0;
}

void usb_dbg_uart_send(const char *buf, unsigned int len) {
  if (!buf || len == 0) return;

  /* clamp len to buffer capacity; if too large, keep the last part */
  if (len > USB_DBG_UART_RING_SZ - 4U) {
    /* keep last (ring size - 4) bytes */
    buf += (len - (USB_DBG_UART_RING_SZ - 4U));
    len = (USB_DBG_UART_RING_SZ - 4U);
  }

  /* Copy into ring buffer (disable IRQ briefly to protect indices) */
  __disable_irq();
  uint32_t free = usb_dbg_uart_ring_free();
  if (len > free) {
    /* Not enough room: drop oldest bytes to make room */
    uint32_t drop = len - free;
    uart_ring_tail = (uart_ring_tail + drop) % USB_DBG_UART_RING_SZ;
  }

  for (unsigned int i = 0; i < len; ++i) {
    usb_dbg_uart_ring[uart_ring_head] = (uint8_t)buf[i];
    uart_ring_head = (uart_ring_head + 1) % USB_DBG_UART_RING_SZ;
  }

  /* If idle, kick off a TX. We can call start_tx with IRQ disabled but
   * HAL transmit routines expect interrupts enabled, so enable before call.
   */
  __enable_irq();

  /* Try to start TX if idle */
  (void)usb_dbg_uart_start_tx_if_idle();
}

/* Callback from HAL when a Tx completes (DMA or IT). We must be prepared
 * to be called from IRQ context. Advance tail and start next chunk if data remains.
 */
void HAL_UART_TxCpltCallback(UART_HandleTypeDef *huart) {
  if (huart != &huart2) return;

  /* advance tail by last sent, protecting indices */
  __disable_irq();
  uart_ring_tail = (uart_ring_tail + uart_last_sent) % USB_DBG_UART_RING_SZ;
  uart_last_sent = 0;
  /* check if more data remains */
  uint32_t used = usb_dbg_uart_ring_used();
  if (used == 0) {
    uart_tx_busy = 0;
    __enable_irq();
    return;
  }
  /* prepare next contiguous chunk length */
  uint32_t t = uart_ring_tail;
  uint32_t cont = (t + used <= USB_DBG_UART_RING_SZ) ? used : (USB_DBG_UART_RING_SZ - t);
  uart_last_sent = cont;
  __enable_irq();

  /* start next transfer (prefer DMA) */
  if (huart2.hdmatx != NULL) {
    if (HAL_UART_Transmit_DMA(&huart2, &usb_dbg_uart_ring[t], (uint16_t)cont) != HAL_OK) {
      (void)HAL_UART_Transmit_IT(&huart2, &usb_dbg_uart_ring[t], (uint16_t)cont);
    }
  } else {
    (void)HAL_UART_Transmit_IT(&huart2, &usb_dbg_uart_ring[t], (uint16_t)cont);
  }
}

void HAL_UART_ErrorCallback(UART_HandleTypeDef *huart) {
  if (huart != &huart2) return;
  /* On error, clear busy so future sends can re-start. */
  uart_tx_busy = 0;
  uart_last_sent = 0;
}

/* USER CODE END USB_DBG_UART */
