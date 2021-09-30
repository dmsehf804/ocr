package com.kakaovx.ocrlib;

import java.lang.reflect.Array;
import java.util.ArrayList;

import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.PointF;
import android.graphics.Canvas;
import android.graphics.RectF;

public class Plot {
    Paint rectPaint;

    public Plot() {
        rectPaint = new Paint();
        rectPaint.setStyle(Paint.Style.STROKE);
    }

    public boolean drawRect(Canvas canvas, ArrayList<PointF> points, float lineWidth, int lineColor) {
        if (points == null || points.size() != 4)
            return false;

        ArrayList<PointF> rect = ImageUtils.alignRect(points);
        if (rect == null)
            return false;

        Path path = new Path();
        path.moveTo(rect.get(0).x, rect.get(0).y);
        path.lineTo(rect.get(1).x, rect.get(1).y);
        path.lineTo(rect.get(2).x, rect.get(2).y);
        path.lineTo(rect.get(3).x, rect.get(3).y);
        path.lineTo(rect.get(0).x, rect.get(0).y);

        rectPaint.setStrokeWidth(lineWidth);
        rectPaint.setColor(lineColor);

        canvas.drawPath(path, rectPaint);

        return true;
    }
}
