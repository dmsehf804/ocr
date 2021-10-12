package com.kakaovx.ocrlib;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Point;
import android.graphics.PointF;
import android.graphics.RectF;

import java.util.ArrayList;

public class ImageUtils {
    static public ArrayList<PointF> alignRect(ArrayList<PointF> points) {
        if (points == null || points.size() != 4)
            return points;

        PointF center = new PointF();

        for (int i = 0; i < points.size(); i++) {
            center.x += points.get(i).x;
            center.y += points.get(i).y;
        }

        center.x /= (float)points.size();
        center.y /= (float)points.size();

        PointF leftTop = null;
        PointF rightTop = null;
        PointF leftBottom = null;
        PointF rightBottom = null;


        for (int i = 0; i < points.size(); i++) {
            PointF point = points.get(i);

            if (point.x < center.x && point.y < center.y)
                leftTop = point;
            else if (point.x < center.x && point.y > center.y)
                leftBottom = point;
            else if (point.x > center.x && point.y < center.y)
                rightTop = point;
            else if (point.x > center.x && point.y > center.y)
                rightBottom = point;
        }

        if (leftTop == null || leftBottom == null || rightTop == null || rightBottom == null)
            return points;

        ArrayList<PointF> rects = new ArrayList<>();

        rects.add(leftTop);
        rects.add(rightTop);
        rects.add(rightBottom);
        rects.add(leftBottom);

        return rects;
    }

    static public RectF getBoundingBox(ArrayList<PointF> points) {
        if (points == null || points.size() != 4)
            return null;

        float min_x = 100000;
        float min_y = 100000;
        float max_x = -1;
        float max_y = -1;

        for (int i = 0; i < points.size(); i++) {
            PointF point = points.get(i);

            min_x = Math.min(min_x, point.x);
            min_y = Math.min(min_y, point.y);
            max_x = Math.max(max_x, point.x);
            max_y = Math.max(max_y, point.y);
            System.out.println("51241234124124");
        }

        return new RectF(min_x, min_y, max_x, max_y);
    }

    static private PointF getRectSize(ArrayList<PointF> points) {
        if (points == null || points.size() != 4)
            return null;

        PointF center = new PointF();

        for (int i = 0; i < points.size(); i++) {
            center.x += points.get(i).x;
            center.y += points.get(i).y;
        }

        center.x /= (float)points.size();
        center.y /= (float)points.size();

        PointF leftTop = null;
        PointF rightTop = null;
        PointF leftBottom = null;
        PointF rightBottom = null;

        for (int i = 0; i < points.size(); i++) {
            PointF point = points.get(i);

            if (point.x < center.x && point.y < center.y)
                leftTop = point;
            else if (point.x < center.x && point.y > center.y)
                leftBottom = point;
            else if (point.x > center.x && point.y < center.y)
                rightTop = point;
            else if (point.x > center.x && point.y > center.y)
                rightBottom = point;
        }

        if (leftTop == null || leftBottom == null || rightTop == null || rightBottom == null)
            return null;

        float width = Math.max(
                rightTop.x - leftTop.x,
                rightBottom.x - leftBottom.x
        );
        float height = Math.max(
                leftBottom.y - leftTop.y,
                rightBottom.y - rightTop.y
        );

        return new PointF(width, height);
    }

    static public Bitmap perspectiveTransform(Bitmap source, ArrayList<PointF> points) {

        float[] src = new float[8];

        PointF point = points.get(0);
        src[0] = point.x;
        src[1] = point.y;

        point = points.get(1);
        src[2] = point.x;
        src[3] = point.y;

        point = points.get(2);
        src[4] = point.x;
        src[5] = point.y;

        point = points.get(3);
        src[6] = point.x;
        src[7] = point.y;

        PointF rectSize = getRectSize(points);
        if (rectSize == null)
            return null;

        // set up a dest polygon which is just a rectangle
        float[] dst = new float[8];
        dst[0] = 0;
        dst[1] = 0;
        dst[2] = rectSize.x;
        dst[3] = 0;
        dst[4] = rectSize.x;
        dst[5] = rectSize.y;
        dst[6] = 0;
        dst[7] = rectSize.y;

        // create a matrix for transformation.
        Matrix matrix = new Matrix();

        // set the matrix to map the source values to the dest values.
        boolean mapped = matrix.setPolyToPoly (src, 0, dst, 0, 4);

        Bitmap newBitmap = Bitmap.createBitmap((int)rectSize.x, (int)rectSize.y, Bitmap.Config.ARGB_8888);
        Canvas newBitmapCanvas = new Canvas(newBitmap);

        newBitmapCanvas.drawBitmap(source, matrix, null);

        return newBitmap;

        //return Bitmap.createBitmap(source, 0, 0, (int)rectSize.x, (int)rectSize.y, matrix, true);


        /*
        float[] mappedTL = new float[] { 0, 0 };
        matrix.mapPoints(mappedTL);
        int maptlx = Math.round(mappedTL[0]);
        int maptly = Math.round(mappedTL[1]);

        float[] mappedTR = new float[] { source.getWidth(), 0 };
        matrix.mapPoints(mappedTR);
        int maptry = Math.round(mappedTR[1]);

        float[] mappedLL = new float[] { 0, source.getHeight() };
        matrix.mapPoints(mappedLL);
        int mapllx = Math.round(mappedLL[0]);

        int shiftX = Math.max(-maptlx, -mapllx);
        int shiftY = Math.max(-maptry, -maptly);

        Bitmap croppedAndCorrected = null;
        if (mapped) {
            Bitmap imageOut = Bitmap.createBitmap(source, 0, 0, source.getWidth(), source.getHeight(), matrix, true);
            croppedAndCorrected = Bitmap.createBitmap(imageOut, shiftX, shiftY, (int)rectSize.x, (int)rectSize.y, null, true);
            imageOut.recycle();
            System.gc();
        }

        return croppedAndCorrected;
         */
    }

    static public Bitmap cropBitmap(final Bitmap source, RectF boundingBox) {
        if (source == null || boundingBox == null)
            return source;

        System.out.println(source.getWidth());
        System.out.println(boundingBox.height());
        System.out.println(boundingBox.width());
        return Bitmap.createBitmap(source, (int) boundingBox.left, (int) boundingBox.top, (int) boundingBox.width(), (int) boundingBox.height());


    }
}
