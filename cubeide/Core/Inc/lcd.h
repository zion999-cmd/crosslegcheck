/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file           : lcd.h
 * @brief          : LCD driver header file
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

#ifndef __LCD_H
#define __LCD_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include <stdint.h>
#include "stm32h7xx_hal.h"

// LCD颜色定义 (16位RGB565格式)
#define RED 0xF800
#define GREEN 0x07E0
#define BLUE 0x001F
#define WHITE 0xFFFF
#define BLACK 0x0000
#define YELLOW 0xFFE0
#define CYAN 0x07FF      // 青色 (Green + Blue)
#define GRAY0 0xEF7D     // 灰色0
#define GRAY1 0x8410     // 灰色1
#define GRAY2 0x4208     // 灰色2
#define DARK_BLUE 0x0010 // 深蓝色，用作背景

/* USER CODE BEGIN PD */
#define LCD_CS_PIN GPIO_PIN_12 // PB12
#define LCD_CS_PORT GPIOB
#define LCD_RST_PIN GPIO_PIN_14 // PB14
#define LCD_RST_PORT GPIOB
#define LCD_DC_PIN GPIO_PIN_1 // PB1
#define LCD_DC_PORT GPIOB
#define LCD_BLK_PIN GPIO_PIN_0 // PB0
#define LCD_BLK_PORT GPIOB

// Software SPI pins
#define LCD_SCK_PIN GPIO_PIN_13 // PB13
#define LCD_SCK_PORT GPIOB
#define LCD_MOSI_PIN GPIO_PIN_15 // PB15
#define LCD_MOSI_PORT GPIOB
/* USER CODE END PD */

extern SPI_HandleTypeDef hspi2;

// 基础LCD函数
void LCD_Reset(void);
void LCD_WriteCommand(uint8_t cmd);
void LCD_WriteData(uint8_t data);
void LCD_Init(void);
void LCD_SetAddrWindow(uint16_t x0, uint16_t y0, uint16_t x1, uint16_t y1);
void LCD_Clear(uint16_t color);
void LCD_FillRect(uint16_t x, uint16_t y, uint16_t w, uint16_t h, uint16_t color);

// 绘图函数
void LCD_DrawPixel(uint16_t x, uint16_t y, uint16_t color);
void Gui_DrawPoint(uint16_t x, uint16_t y, uint16_t color);

// 文字显示函数
void LCD_DrawNumber(uint16_t x, uint16_t y, uint32_t number, uint16_t color, uint16_t bg_color);
void LCD_DrawChar(uint16_t x, uint16_t y, char ch, uint16_t color, uint16_t bg_color);
void LCD_DrawString(uint16_t x, uint16_t y, const char *str, uint16_t color, uint16_t bg_color);

// 测试函数
void LCD_TestFont(uint32_t current_time);

#ifdef __cplusplus
}
#endif

#endif /* __LCD_H */