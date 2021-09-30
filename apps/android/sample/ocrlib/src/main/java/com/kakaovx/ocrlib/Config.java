package com.kakaovx.ocrlib;

public class Config {
    public static int input_width = 160;
    public static int input_height = 160;
    public static int input_channel = 3;

    public static int output_stride = 2;

    public static int output_width = input_width / output_stride;
    public static int output_height = input_height / output_stride;

    public static float heatmap_threshold = 0.3f;
    public static int local_maximum_radius = 4;
}
