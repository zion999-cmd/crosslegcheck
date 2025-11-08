/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file           : main.c
 * @brief          : Main program body - Clean USB Host CDC
 ******************************************************************************
 */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "lcd.h"
#include "usb_host.h"
#include "usbh_cdc.h"
#include "usbh_core.h"
#include "usb_dbg.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
/* USER CODE END Includes */

/* Private variables ---------------------------------------------------------*/
SPI_HandleTypeDef hspi2;
UART_HandleTypeDef huart2;

/* USER CODE BEGIN PV */
// DMA传输状态标志 - LCD需要
volatile uint8_t dma_transfer_complete = 1;

// CDC数据缓冲区
uint8_t CDC_RX_Buffer[512];

// USB Host相关变量
extern USBH_HandleTypeDef hUsbHostFS;
extern ApplicationTypeDef Appli_state;
// --- CDC 流处理 (环形缓冲, 在 callback 写入, 主循环解析) ---
#define CDC_STREAM_BUF_SIZE 1024
uint8_t cdc_stream_buf[CDC_STREAM_BUF_SIZE];
volatile uint32_t cdc_stream_head = 0; // 写指针
volatile uint32_t cdc_stream_tail = 0; // 读指针
volatile uint8_t cdc_data_available = 0; // 表示有新数据
volatile float last_pressure = 0.0f; // 最近解析出的压力值
// 额外诊断变量
volatile uint32_t last_rx_len = 0; // 上一次回调收到的字节数
char last_rx_preview[64] = {0}; // 回显最近接收到的数据（预览）
char last_line[128] = {0}; // 最后解析出的完整行文本
volatile uint32_t last_line_time_ms = 0; // 上次成功解析行的时间戳 (ms)
// 解析得到的最近一行的数值（最多 12 个传感器）
float last_values[12] = {0.0f};
int last_values_count = 0;
volatile uint8_t last_values_ready = 0; // 标志：最近一行的值可用

// 解析状态：用于主循环调用
void process_cdc_stream(void);
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MPU_Config(void);
static void MX_GPIO_Init(void);
void MX_USB_HOST_Init(void);
void MX_USB_HOST_Process(void);
void MX_USART2_UART_Init(void);

void CDC_DebugInfo_Display(void);

// 简化显示逻辑：去掉复杂的显示模式开关，屏幕只用于日志和 CDC 状态
// HUD 行缓存和简单行绘制函数
/* Use more lines to occupy the full 240px height (font 8x16 -> 15 lines max).
 * Use 14 to leave a small margin. */
/* USER CODE BEGIN HUD */
#define HUD_LINES 14
static char hud_last[HUD_LINES][64];
static void hud_draw_line(int idx, uint16_t y, const char *fmt, ...) {
  if (idx < 0 || idx >= HUD_LINES) return;
  char buf[64];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  // 固定宽度填充，确保短文本会覆盖旧的长文本残留
  int len = strlen(buf);
  if (len < 32) {
    memset(buf + len, ' ', 32 - len);
    buf[32] = '\0';
  }
  if (strncmp(hud_last[idx], buf, sizeof(buf)) != 0) {
    // 保存并绘制
    strncpy(hud_last[idx], buf, sizeof(hud_last[idx]) - 1);
    hud_last[idx][sizeof(hud_last[idx]) - 1] = '\0';
    LCD_DrawString(0, y, buf, WHITE, DARK_BLUE);
  }
}
/* USER CODE END HUD */

/* USER CODE BEGIN PFP */
// 是否启用 USART2 手动 DMA
#define ENABLE_USART2_MANUAL_DMA
/* USER CODE END PFP */

/**
 * @brief  The application entry point.
 * @retval int
 */
int main(void) {
  /* USER CODE BEGIN 1 */
  /* USER CODE END 1 */

  /* MPU Configuration--------------------------------------------------------*/
  MPU_Config();

  /* MCU Configuration--------------------------------------------------------*/
  HAL_Init();

  /* USER CODE BEGIN Init */
  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */
  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_SPI2_Init();
  MX_USB_HOST_Init();
  /* Keep runtime USBH logging disabled by default (we only want raw
   * USB->UART passthrough). If you need logs later enable them at runtime. */
  extern void usbh_enable_runtime_logging(uint8_t enable);
  /* 关闭运行时日志，保持仅做透明的 USB->UART 数据透传（不输出调试日志）。 */
  usbh_enable_runtime_logging(0);
  usb_dbg_set_log_mask(0);
  /* Initialize USART2 for log forwarding (PA2=TX, PA3=RX) */
  MX_USART2_UART_Init();

  /* Quick UART send test: enqueue a short test line to verify TX path
   * (uses non-blocking usb_dbg_uart_send -> IT/DMA). This helps confirm
   * wiring and that the IRQ/IT fallback works even if DMA isn't configured.
   */
  HAL_Delay(50);

  /* USER CODE BEGIN 2 */
  // Initialize LCD
  HAL_GPIO_WritePin(LCD_BLK_PORT, LCD_BLK_PIN, GPIO_PIN_SET);
  HAL_Delay(100);

  LCD_Reset();
  LCD_Init();
  LCD_Clear(DARK_BLUE);

  /* Title removed to keep full screen for logs */
  
  // LED指示启动完成
  HAL_GPIO_WritePin(GPIOA, GPIO_PIN_1, GPIO_PIN_SET);
  HAL_Delay(200);
  HAL_GPIO_WritePin(GPIOA, GPIO_PIN_1, GPIO_PIN_RESET);
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1) {

        /* USER CODE BEGIN 3 */
    // 获取当前时间
    uint32_t current_time = HAL_GetTick();
    
    /* Uptime display removed to avoid frequent LCD updates (reduces CPU load)
       and keep the whole screen available for logs */
    
    // USB Host处理
    MX_USB_HOST_Process();
    

    
    // 优先处理 CDC 环形缓冲（若有数据）以保持及时透传
    if (cdc_data_available) {
      process_cdc_stream();
    }

    // 状态显示 - 每200ms更新（不影响 CDC 转发）
    static uint32_t last_status_update = 0;
    if (current_time - last_status_update >= 200) {
      last_status_update = current_time;
      CDC_DebugInfo_Display();
    }

    // Keep periodic test sends disabled during CDC debug to avoid noise
    (void)current_time;

  /* USER CODE END WHILE */
  }
  /* USER CODE END 3 */
}

/* USER CODE BEGIN CDC_DEBUG_DISPLAY */
// CDC调试信息显示功能  
void CDC_DebugInfo_Display(void) {
  /* Minimal HUD: show a single heartbeat/status line and avoid frequent redraws.
   * This reduces LCD activity which previously affected USB timing. Detailed
   * logs are forwarded via USART2 (Bluetooth) only.
   */
  static uint32_t prev_state = 0xFFFFFFFFU;
  static uint32_t last_draw = 0;
  uint32_t now = HAL_GetTick();

  /* Throttle to ~1s updates */
  if (now - last_draw < 1000 && (uint32_t)Appli_state == prev_state) {
    return;
  }
  last_draw = now;
  prev_state = (uint32_t)Appli_state;

  /* Keep runtime logging disabled even when ready to avoid log noise. */

  /* Single line status at bottom */
  const int start_y = (HUD_LINES - 1) * 16;
  hud_draw_line(HUD_LINES - 1, start_y, "Alive CDC:%s H:%d E:%d A:%d",
                (Appli_state == APPLICATION_READY) ? "Ready" : "NotRd",
                (int)hUsbHostFS.gState,
                (int)hUsbHostFS.EnumState,
                (int)Appli_state);
}
/* USER CODE END CDC_DEBUG_DISPLAY */

/* USER CODE BEGIN CDC_HANDLERS */
// CDC接收回调函数
void USBH_CDC_ReceiveCallback(USBH_HandleTypeDef *phost) {
  uint32_t rx_len = USBH_CDC_GetLastReceivedDataSize(phost);
  if (rx_len > 0) {
    /* Fast path: copy received bytes into local ring buffer and re-arm
     * the USB receive before any UART work. This avoids blocking the USB
     * callback while allowing the main loop to forward data to UART.
     */
    for (uint32_t i = 0; i < rx_len; i++) {
      uint32_t next = (cdc_stream_head + 1) % CDC_STREAM_BUF_SIZE;
      if (next == cdc_stream_tail) {
        /* Buffer full: drop oldest byte to make room */
        cdc_stream_tail = (cdc_stream_tail + 1) % CDC_STREAM_BUF_SIZE;
      }
      cdc_stream_buf[cdc_stream_head] = CDC_RX_Buffer[i];
      cdc_stream_head = next;
    }
    cdc_data_available = 1;
    /* Re-arm USB receive quickly */
    USBH_CDC_Receive(phost, CDC_RX_Buffer, sizeof(CDC_RX_Buffer));
  }
}

// 解析环形缓冲中的数据，寻找以 '\n' 结尾的行并尝试解析为浮点压力值
void process_cdc_stream(void) {
  /* Drain a small chunk from the ring buffer and forward it unchanged to
   * the UART via the non-blocking usb_dbg_uart_send(). Limit chunk size to
   * avoid long blocking in the main loop. */
  const uint32_t CHUNK_SZ = 64;
  uint8_t tmp[CHUNK_SZ];
  uint32_t copied = 0;

  while (cdc_stream_tail != cdc_stream_head && copied < CHUNK_SZ) {
    tmp[copied++] = cdc_stream_buf[cdc_stream_tail];
    cdc_stream_tail = (cdc_stream_tail + 1) % CDC_STREAM_BUF_SIZE;
  }

  if (copied > 0) {
    extern void usb_dbg_uart_send(const char *buf, unsigned int len);
    usb_dbg_uart_send((const char *)tmp, copied);
  }


  cdc_data_available = (cdc_stream_tail != cdc_stream_head) ? 1 : 0;
}
/* USER CODE END CDC_HANDLERS */


// 系统配置函数
void SystemClock_Config(void) {
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  HAL_PWREx_ConfigSupply(PWR_LDO_SUPPLY);
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE3);

  while (!__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY)) {
  }

  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI48 | RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_DIV1;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.HSI48State = RCC_HSI48_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 4;
  RCC_OscInitStruct.PLL.PLLN = 10;
  RCC_OscInitStruct.PLL.PLLP = 2;
  RCC_OscInitStruct.PLL.PLLQ = 2;
  RCC_OscInitStruct.PLL.PLLR = 2;
  RCC_OscInitStruct.PLL.PLLRGE = RCC_PLL1VCIRANGE_3;
  RCC_OscInitStruct.PLL.PLLVCOSEL = RCC_PLL1VCOMEDIUM;
  RCC_OscInitStruct.PLL.PLLFRACN = 0;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK) {
    Error_Handler();
  }

  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK |
                                RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2 |
                                RCC_CLOCKTYPE_D3PCLK1 | RCC_CLOCKTYPE_D1PCLK1;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_HSI;
  RCC_ClkInitStruct.SYSCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB3CLKDivider = RCC_APB3_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_APB1_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_APB2_DIV2;
  RCC_ClkInitStruct.APB4CLKDivider = RCC_APB4_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_1) != HAL_OK) {
    Error_Handler();
  }
}

void MX_SPI2_Init(void) {
  hspi2.Instance = SPI2;
  hspi2.Init.Mode = SPI_MODE_MASTER;
  hspi2.Init.Direction = SPI_DIRECTION_2LINES;
  hspi2.Init.DataSize = SPI_DATASIZE_8BIT;
  hspi2.Init.CLKPolarity = SPI_POLARITY_HIGH;
  hspi2.Init.CLKPhase = SPI_PHASE_2EDGE;
  hspi2.Init.NSS = SPI_NSS_SOFT;
  hspi2.Init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_16;
  hspi2.Init.FirstBit = SPI_FIRSTBIT_MSB;
  hspi2.Init.TIMode = SPI_TIMODE_DISABLE;
  hspi2.Init.CRCCalculation = SPI_CRCCALCULATION_DISABLE;
  hspi2.Init.CRCPolynomial = 0x0;
  hspi2.Init.NSSPMode = SPI_NSS_PULSE_ENABLE;
  hspi2.Init.NSSPolarity = SPI_NSS_POLARITY_LOW;
  hspi2.Init.FifoThreshold = SPI_FIFO_THRESHOLD_01DATA;
  hspi2.Init.TxCRCInitializationPattern = SPI_CRC_INITIALIZATION_ALL_ZERO_PATTERN;
  hspi2.Init.RxCRCInitializationPattern = SPI_CRC_INITIALIZATION_ALL_ZERO_PATTERN;
  hspi2.Init.MasterSSIdleness = SPI_MASTER_SS_IDLENESS_00CYCLE;
  hspi2.Init.MasterInterDataIdleness = SPI_MASTER_INTERDATA_IDLENESS_00CYCLE;
  hspi2.Init.MasterReceiverAutoSusp = SPI_MASTER_RX_AUTOSUSP_DISABLE;
  hspi2.Init.MasterKeepIOState = SPI_MASTER_KEEP_IO_STATE_DISABLE;
  hspi2.Init.IOSwap = SPI_IO_SWAP_DISABLE;
  if (HAL_SPI_Init(&hspi2) != HAL_OK) {
    Error_Handler();
  }
}

/**
  * @brief USART2 Initialization Function
  * @param None
  * @retval None
  */
void MX_USART2_UART_Init(void)
{

  /* USER CODE BEGIN USART2_Init 0 */

  /* USER CODE END USART2_Init 0 */

  /* USER CODE BEGIN USART2_Init 1 */

  /* USER CODE END USART2_Init 1 */
  huart2.Instance = USART2;
  huart2.Init.BaudRate = 9600;
  huart2.Init.WordLength = UART_WORDLENGTH_8B;
  huart2.Init.StopBits = UART_STOPBITS_1;
  huart2.Init.Parity = UART_PARITY_NONE;
  huart2.Init.Mode = UART_MODE_TX_RX;
  huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart2.Init.OverSampling = UART_OVERSAMPLING_16;
  huart2.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart2.Init.ClockPrescaler = UART_PRESCALER_DIV1;
  huart2.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart2) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetTxFifoThreshold(&huart2, UART_TXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetRxFifoThreshold(&huart2, UART_RXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_DisableFifoMode(&huart2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART2_Init 2 */

  /* USER CODE END USART2_Init 2 */

}

static void MX_GPIO_Init(void) {
  GPIO_InitTypeDef GPIO_InitStruct = {0};

  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  HAL_GPIO_WritePin(GPIOA, GPIO_PIN_1, GPIO_PIN_RESET);
  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0 | GPIO_PIN_1 | GPIO_PIN_12, GPIO_PIN_RESET);

  GPIO_InitStruct.Pin = GPIO_PIN_1;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = GPIO_PIN_0 | GPIO_PIN_1 | GPIO_PIN_12;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  HAL_SYSCFG_AnalogSwitchConfig(SYSCFG_SWITCH_PA1, SYSCFG_SWITCH_PA1_CLOSE);
}

void MPU_Config(void) {
  MPU_Region_InitTypeDef MPU_InitStruct = {0};

  HAL_MPU_Disable();

  MPU_InitStruct.Enable = MPU_REGION_ENABLE;
  MPU_InitStruct.Number = MPU_REGION_NUMBER0;
  MPU_InitStruct.BaseAddress = 0x0;
  MPU_InitStruct.Size = MPU_REGION_SIZE_4GB;
  MPU_InitStruct.SubRegionDisable = 0x87;
  MPU_InitStruct.TypeExtField = MPU_TEX_LEVEL0;
  MPU_InitStruct.AccessPermission = MPU_REGION_NO_ACCESS;
  MPU_InitStruct.DisableExec = MPU_INSTRUCTION_ACCESS_DISABLE;
  MPU_InitStruct.IsShareable = MPU_ACCESS_SHAREABLE;
  MPU_InitStruct.IsCacheable = MPU_ACCESS_NOT_CACHEABLE;
  MPU_InitStruct.IsBufferable = MPU_ACCESS_NOT_BUFFERABLE;

  HAL_MPU_ConfigRegion(&MPU_InitStruct);
  HAL_MPU_Enable(MPU_PRIVILEGED_DEFAULT);
}

void Error_Handler(void) {
  __disable_irq();
  while (1) {
  }
}

