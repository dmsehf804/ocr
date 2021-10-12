package com.kakaovx.ocrlib;

import android.app.Application;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.PointF;
import android.graphics.RectF;
import android.os.SystemClock;
import android.util.Pair;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.commons.math3.util.CombinatoricsUtils;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import java.io.FileInputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

public class ScoreCardOCR {
    /** Options for configuring the Interpreter. */
    private final Interpreter.Options tfliteOptions = new Interpreter.Options();

    /** An instance of the driver class to run model inference with Tensorflow Lite. */
    protected Interpreter tflite;

    private Bitmap croppedBitmap = null;
    private Canvas croppedBitmapCanvas = null;

    private Matrix frameToCropTransform = new Matrix();

    private float inputScale = 1.0f;

    private CommHelper commHelper;
    private Plot plot;

    private ArrayList<OneEuroFilter> oneEuroFilters;


    public ScoreCardOCR(Application application, String model_file) {
        commHelper = new CommHelper();
        plot = new Plot();

    }

    public Bitmap cropScorecardBitmap(
            final Bitmap bitmap,
            final RectF boundingBox) {
        if (bitmap == null)
            return null;

        Bitmap transform = ImageUtils.cropBitmap(bitmap, boundingBox);


        return transform;

    }

    public ScoreCard sendScorecard(
            final Bitmap bitmap,
            final String userId,
            final String userName) {


        try {
            System.out.println("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
            return commHelper.sendScorecard(bitmap, userId, userName);
            //return commHelper.sendScorecardDummy(bitmapScorecard, userId, userName);
        }
        catch (Exception ex) {
            ex.printStackTrace();

            ScoreCard scoreCard = new ScoreCard();
            scoreCard.resultCode = 0;
            scoreCard.resultMessage = ex.getMessage();

            return scoreCard;
        }
    }

    public boolean drawScorecardBox(Canvas canvas, ArrayList<PointF> points, float lineWidth, int lineColor) {
        if (canvas == null)
            return false;

        return plot.drawRect(canvas, points, lineWidth, lineColor);
    }

    public boolean isRotated(int sensorOrientation) {
        return (Math.abs(sensorOrientation) + 90) % 180 == 0;
    }
}
