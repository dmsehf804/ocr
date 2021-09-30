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
    private int[] intValues;

    protected ByteBuffer tfliteFloatValues = null;
    protected float[][][][] tfliteHeatmaps = null;

    private float inputScale = 1.0f;

    private CommHelper commHelper;
    private Plot plot;

    private ArrayList<OneEuroFilter> oneEuroFilters;

    class PointFWithScore {
        PointF point;
        float score;

        public PointFWithScore(float x, float y, float score) {
            this.score = score;
            this.point = new PointF(x, y);
        }
    }

    public ScoreCardOCR(Application application, String model_file) {
        tfliteFloatValues = ByteBuffer.allocateDirect(Config.input_height * Config.input_width * Config.input_channel * 4);
        tfliteFloatValues.order(ByteOrder.nativeOrder());

        tfliteHeatmaps = new float[1][Config.output_height][Config.output_width][1];

        initializeTFLite(application, model_file);

        commHelper = new CommHelper();
        plot = new Plot();

        initializeOneEuroFilters();
    }

    public void release() {
        if (tflite != null)
            tflite.close();

        commHelper = null;
        plot = null;

        croppedBitmap = null;
        croppedBitmapCanvas = null;

        intValues = null;

        tfliteFloatValues = null;
        tfliteHeatmaps = null;

        oneEuroFilters = null;
    }

    private MappedByteBuffer loadModelFile(String path, AssetManager assetManager) {
        try {
            AssetFileDescriptor fileDescriptor = assetManager.openFd(path);
            FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
            return inputStream.getChannel().map(
                    FileChannel.MapMode.READ_ONLY, fileDescriptor.getStartOffset(), fileDescriptor.getDeclaredLength()
            );
        }
        catch (Exception ex) {
            ex.printStackTrace();
            return null;
        }
    }

    private boolean initializeTFLite(Application application, String model_file) {
        CompatibilityList compatList = new CompatibilityList();

        if(compatList.isDelegateSupportedOnThisDevice()){
            // if the device has a supported GPU, add the GPU delegate
            GpuDelegate.Options delegateOptions = compatList.getBestOptionsForThisDevice();
            GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions);
            tfliteOptions.addDelegate(gpuDelegate);
        } else {
            // if the GPU is not supported, run on 4 threads
            tfliteOptions.setNumThreads(4);
        }

        try {
            tflite = new Interpreter(loadModelFile(model_file, application.getAssets()), tfliteOptions);
        }
        catch (Exception ex) {
            ex.printStackTrace();
            return false;
        }

        return true;
    }

    /**
     * Returns a transformation matrix from one reference frame into another. Handles cropping (if maintaining aspect
     * ratio is desired) and rotation.
     *
     * @param srcWidth Width of source frame.
     * @param srcHeight Height of source frame.
     * @param dstWidth Width of destination frame.
     * @param dstHeight Height of destination frame.
     * @param applyRotation Amount of rotation to apply from one frame to another. Must be a multiple of 90.
     * @param maintainAspectRatio If true, will ensure that scaling in x and y remains constant, cropping the image if
     * necessary.
     * @return The transformation fulfilling the desired requirements.
     */
    private Matrix getTransformationMatrix(
            final Matrix matrix,
            final int srcWidth,
            final int srcHeight,
            final int dstWidth,
            final int dstHeight,
            final int applyRotation,
            final boolean maintainAspectRatio) {
        matrix.reset();

        // Translate so center of image is at origin.
        matrix.postTranslate(-srcWidth / 2.0f, -srcHeight / 2.0f);

        if (applyRotation != 0) {
            // Rotate around origin.
            matrix.postRotate(applyRotation);
        }

        // Account for the already applied rotation, if any, and then determine how
        // much scaling is needed for each axis.
        final boolean transpose = (Math.abs(applyRotation) + 90) % 180 == 0;

        final int inWidth = transpose ? srcHeight : srcWidth;
        final int inHeight = transpose ? srcWidth : srcHeight;

        // Apply scaling if necessary.
        if (inWidth != dstWidth || inHeight != dstHeight) {
            final float scaleFactorX = dstWidth / (float) inWidth;
            final float scaleFactorY = dstHeight / (float) inHeight;

            if (maintainAspectRatio) {
                // Scale by minimum factor so that dst is filled completely while
                // maintaining the aspect ratio. Some image may fall off the edge.
                final float scaleFactor = Math.max(scaleFactorX, scaleFactorY);
                matrix.postScale(scaleFactor, scaleFactor);
            } else {
                // Scale exactly to fill dst from src.
                matrix.postScale(scaleFactorX, scaleFactorY);
            }
        }

        matrix.postTranslate(dstWidth / 2.0f, dstHeight / 2.0f);

        /*
        if (applyRotation != 0) {
            // Translate back from origin centered reference to destination frame.
            matrix.postTranslate(dstWidth / 2.0f, dstHeight / 2.0f);
        }
        */

        return matrix;
    }

    private Matrix getTransformationMatrix(final int srcWidth,
                                          final int srcHeight,
                                          final int applyRotation,
                                          final boolean flip) {
        int targetWidth = Config.input_width;
        int targetHeight = Config.input_height;

        inputScale = 1.0f;
        float scaleCenterX = targetWidth / 2.0f;
        float scaleCenterY = targetHeight / 2.0f;

        getTransformationMatrix(
                frameToCropTransform,
                srcWidth, srcHeight,
                targetWidth, targetHeight,
                applyRotation, true);

        if (flip)
            frameToCropTransform.postScale(-1, 1, targetWidth / 2, targetHeight / 2);

        if (inputScale != 1.0f) {
            frameToCropTransform.postTranslate(targetWidth / 2 - scaleCenterX, targetHeight / 2 - scaleCenterY);
        }

        return frameToCropTransform;
    }

    public Bitmap getCroppedBitmap() {
        return croppedBitmap;
    }

    public int getInputWidth() {
        return Config.input_width;
    }
    public int getInputHeight() {
        return Config.input_height;
    }

    public ArrayList<PointF> inference(Bitmap bitmap, final int applyRotation, final boolean flip) {
        if (bitmap == null)
            return null;

        if (croppedBitmap == null || croppedBitmapCanvas == null) {
            croppedBitmap = Bitmap.createBitmap(Config.input_width, Config.input_height, Bitmap.Config.ARGB_8888);
            croppedBitmapCanvas = new Canvas(croppedBitmap);
        }

        int originWidth = isRotated(applyRotation) ? bitmap.getHeight() : bitmap.getWidth();
        int originHeight = isRotated(applyRotation) ? bitmap.getWidth() : bitmap.getHeight();

        float scaleRatio = Math.min(originWidth / (float)Config.output_width, originHeight / (float)Config.output_height);

        Matrix transformMatrix = getTransformationMatrix(bitmap.getWidth(), bitmap.getHeight(), applyRotation, flip);
        croppedBitmapCanvas.drawBitmap(bitmap, transformMatrix, null);

        if (intValues == null)
            intValues = new int[Config.input_height * Config.input_width];
        croppedBitmap.getPixels(intValues, 0, croppedBitmap.getWidth(), 0, 0, croppedBitmap.getWidth(), croppedBitmap.getHeight());

        tfliteFloatValues.rewind();
        for (int i = 0; i < intValues.length; ++i) {
            // RGB
            //tfliteFloatValues.putFloat(((intValues[i] >> 16) & 0xFF) / 255.f - 0.5f);
            //tfliteFloatValues.putFloat(((intValues[i] >> 8) & 0xFF) / 255.f - 0.5f);
            //tfliteFloatValues.putFloat((intValues[i] & 0xFF) / 255.f - 0.5f);

            // BGR
            tfliteFloatValues.putFloat((intValues[i] & 0xFF) / 255.f - 0.5f);
            tfliteFloatValues.putFloat(((intValues[i] >> 8) & 0xFF) / 255.f - 0.5f);
            tfliteFloatValues.putFloat(((intValues[i] >> 16) & 0xFF) / 255.f - 0.5f);
        }

        Object[] inputArray = { tfliteFloatValues };

        Map<Integer, Object> outputMap = new HashMap<>();
        outputMap.put(0, tfliteHeatmaps);

        try {
            tflite.runForMultipleInputsOutputs(inputArray, outputMap);
        } catch (Exception ex) {
            ex.printStackTrace();
        }

        if (tfliteHeatmaps.length <= 0)
            return null;

        PriorityQueue<PointFWithScore> queue = computeHeatmaps(tfliteHeatmaps[0], Config.heatmap_threshold);
        //if (queue == null || queue.size() < 4)
        if (queue == null)
            return null;

        ArrayList<PointF> points = new ArrayList<>();

        while (queue.size() > 0) {
            PointFWithScore point = queue.poll();

            float point_x = (point.point.x - Config.output_width / 2) * scaleRatio + originWidth / 2.f;
            float point_y = (point.point.y - Config.output_height / 2) * scaleRatio + originHeight / 2.f;

            points.add(new PointF(point_x, point_y));
        }

        points = ImageUtils.alignRect(filterPoints(points));
        if (points.size() == 4) {
            applyOneEuroFilter(points);
        }

        return points;
    }

    private ArrayList<PointF> filterPoints(ArrayList<PointF> points) {
        if (points == null || points.size() <= 4)
            return points;

        ArrayList<PointF> filtered = new ArrayList<>();

        SummaryStatistics stats = new SummaryStatistics();
        double minStd = -1;

        Iterator<int[]> iterator = CombinatoricsUtils.combinationsIterator(points.size(), 4);
        while (iterator.hasNext()) {
            final int[] combination = iterator.next();

            float sum_x = 0;
            float sum_y = 0;
            for (int i = 0; i < combination.length; i++) {
                sum_x += points.get(combination[i]).x;
                sum_y += points.get(combination[i]).y;
            }

            float center_x = sum_x / (float)combination.length;
            float center_y = sum_y / (float)combination.length;

            stats.clear();

            for (int i = 0; i < combination.length; i++) {
                double dist = Math.sqrt(Math.pow(center_x - points.get(combination[i]).x, 2) + Math.pow(center_y - points.get(combination[i]).y, 2));
                stats.addValue(dist);
            }

            double std = stats.getStandardDeviation();
            if (std > 100)
                continue;

            if (minStd < 0 || std < minStd) {
                minStd = std;
                filtered.clear();
                for (int i = 0; i < combination.length; i++)
                    filtered.add(points.get(combination[i]));
            }
        }

        return filtered;
    }

    private boolean initializeOneEuroFilters() {
        if (oneEuroFilters == null)
            oneEuroFilters = new ArrayList<>();
        else
            oneEuroFilters.clear();

        for (int i = 0; i < 4 * 2; i++) {
            try {
                OneEuroFilter filter = new OneEuroFilter(30, 1.0, 0.007, 1.0);
                oneEuroFilters.add(filter);
            } catch (Exception ex) {
                ex.printStackTrace();
                return false;
            }
        }

        return false;
    }

    private boolean applyOneEuroFilter(ArrayList<PointF> points) {
        if (points == null || points.size() != 4)
            return false;

        if (oneEuroFilters == null || oneEuroFilters.size() < 4) {
            if (initializeOneEuroFilters() == false)
                return false;
        }

        double timestamp = SystemClock.uptimeMillis() / 1000.0;

        try {
            for (int i = 0; i < 4; i++) {
                points.get(i).x = (float) oneEuroFilters.get(i).filter(points.get(i).x, timestamp);
                points.get(i).y = (float) oneEuroFilters.get(i + 4).filter(points.get(i).y, timestamp);
            }
        } catch (Exception ex) {
            ex.printStackTrace();
            return false;
        }

        return true;
    }

    private boolean scoreIsMaximumInLocalWindow(float score, int heatmapY, int heatmapX, int localMaximumRadius, float[][][] heatmaps) {
        boolean localMaximum = true;

        int yStart = Math.max(heatmapY - localMaximumRadius, 0);
        int yEnd = Math.min(heatmapY + localMaximumRadius + 1, heatmaps.length);
        for (int yCurrent = yStart; yCurrent < yEnd; ++yCurrent) {
            int xStart = Math.max(heatmapX - localMaximumRadius, 0);
            int xEnd = Math.min(heatmapX + localMaximumRadius + 1, heatmaps[0].length);
            for (int xCurrent = xStart; xCurrent < xEnd; ++xCurrent) {
                if (heatmaps[yCurrent][xCurrent][0] > score) {
                    localMaximum = false;
                    break;
                }
            }

            if (localMaximum == false)
                break;
        }

        return localMaximum;
    }

    private PriorityQueue<PointFWithScore> computeHeatmaps(float[][][] heatmaps, float scoreThreshold) {
        if (heatmaps == null || heatmaps.length < 1 || heatmaps[0].length < 1 || heatmaps[0][0].length < 1)
            return null;

        PriorityQueue<PointFWithScore> queue = new PriorityQueue<>(4, new Comparator<PointFWithScore>() {
            @Override
            public int compare(PointFWithScore o1, PointFWithScore o2) {
                if (o1.score > o2.score)
                    return -1;
                else if (o1.score < o2.score)
                    return 1;
                return 0;
            }
        });

        for (int heatmapY = 0; heatmapY < heatmaps.length; ++heatmapY) {
            for (int heatmapX = 0; heatmapX < heatmaps[0].length; ++heatmapX) {
                float score = heatmaps[heatmapY][heatmapX][0];

                // Only consider parts with score greater or equal to threshold as root candidates.
                if (score < scoreThreshold)
                    continue;

                // Only consider keypoints whose score is maximum in a local window.
                if (scoreIsMaximumInLocalWindow(score, heatmapY, heatmapX, Config.local_maximum_radius, heatmaps)) {
                    queue.add(new PointFWithScore(heatmapX, heatmapY, score));
                }
            }
        }

        return queue;
    }

    public Pair<Bitmap, ArrayList<PointF>> cropScorecardBitmap(
            final Bitmap bitmap,
            final ArrayList<PointF> points) {
        if (bitmap == null)
            return null;

//        int targetWidth =  bitmap.getWidth();
//        int targetHeight = bitmap.getHeight();
//
//        Bitmap rotateBitmap = Bitmap.createBitmap(targetWidth, targetHeight, Bitmap.Config.ARGB_8888);
//        Canvas rotateBitmapCanvas = new Canvas(rotateBitmap);
//
//        float scaleCenterX = targetWidth / 2.0f;
//        float scaleCenterY = targetHeight / 2.0f;
//
//        if (inputScale != 1.0f) {
//            frameToCropTransform.postTranslate(targetWidth / 2 - scaleCenterX, targetHeight / 2 - scaleCenterY);
//        }
//
//        rotateBitmapCanvas.drawBitmap(bitmap, frameToCropTransform, null);
//
//        ArrayList<PointF> rect = ImageUtils.alignRect(points);
//        if (rect == null)
//            return Pair.create(rotateBitmap, points);
//
//
//        RectF boundingBox = ImageUtils.getBoundingBox(rect);
//        if (boundingBox == null)
//            return Pair.create(rotateBitmap, points);
//
//        Bitmap transform = ImageUtils.cropBitmap(rotateBitmap, boundingBox);
//        if (transform == null)
//            return Pair.create(rotateBitmap, points);
//
//        ArrayList<PointF> newPoints = new ArrayList<>();
//        newPoints.add(new PointF(rect.get(0).x - boundingBox.left, rect.get(0).y - boundingBox.top));
//        newPoints.add(new PointF(rect.get(1).x - boundingBox.left, rect.get(1).y - boundingBox.top));
//        newPoints.add(new PointF(rect.get(2).x - boundingBox.left, rect.get(2).y - boundingBox.top));
//        newPoints.add(new PointF(rect.get(3).x - boundingBox.left, rect.get(3).y - boundingBox.top));

        return Pair.create(bitmap, points);

    }

    public ScoreCard sendScorecard(
            final Bitmap bitmap,
            final ArrayList<PointF> points,
            final String userId,
            final String userName) {
        Pair<Bitmap, ArrayList<PointF>> pair = cropScorecardBitmap(bitmap, points);
        if (pair == null || pair.first == null)
            return null;

        try {
            System.out.println("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
            return commHelper.sendScorecard(pair.first, pair.second, userId, userName);
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
