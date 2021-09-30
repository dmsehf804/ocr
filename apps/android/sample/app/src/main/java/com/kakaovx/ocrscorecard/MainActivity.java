package com.kakaovx.ocrscorecard;

import android.app.AlertDialog;
import android.app.Dialog;
import android.app.ProgressDialog;
import android.content.DialogInterface;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.PointF;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.SystemClock;
import android.util.Pair;
import android.util.Size;
import android.util.TypedValue;
import android.view.View;
import android.widget.EditText;
import android.widget.Toast;

import java.util.ArrayList;

import com.kakaovx.ocrlib.ScoreCard;
import com.kakaovx.ocrscorecard.customview.OverlayView;
import com.kakaovx.ocrscorecard.env.BorderedText;
import com.kakaovx.ocrscorecard.env.Logger;

import com.kakaovx.ocrlib.ScoreCardOCR;

public class MainActivity extends CameraActivity implements OnImageAvailableListener {
    private static final Logger LOGGER = new Logger();

    // Minimum detection confidence to track a detection.
    //private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
    private static final Size DESIRED_PREVIEW_SIZE = new Size(1920, 1440);
    private static final float TEXT_SIZE_DIP = 10;
    OverlayView trackingOverlay;
    private Integer sensorOrientation;

    private ScoreCardOCR detector;

    private long lastProcessingTimeMs;
    private Bitmap rgbFrameBitmap = null;
    private final Object lockFrameBitmap = new Object();

    private boolean computingDetection = false;

    private long timestamp = 0;

    private BorderedText borderedText;

    private ArrayList<PointF> points = null;

    private Paint paint = null;
    private final Object lockPoints = new Object();

    private boolean isAffine = false;

    @Override
    protected void onCreateActivity() {
        super.onCreateActivity();

        findViewById(R.id.progressbar).setVisibility(View.GONE);
    }

    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {
        final float textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);

        try {
            detector = new ScoreCardOCR(getApplication(), "detect_corner.tflite");
        } catch (final Exception e) {
            e.printStackTrace();
            LOGGER.e(e, "Exception initializing Detector!");
            Toast toast =
                    Toast.makeText(
                            getApplicationContext(), "Detector could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            finish();
        }

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        sensorOrientation = rotation - getScreenOrientation();
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);

        paint = new Paint();
        paint.setColor(Color.rgb(255, 0, 0));
        paint.setStyle(Paint.Style.FILL);

        trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
        trackingOverlay.addCallback(canvas -> {
            int imageWidth = detector.isRotated(sensorOrientation) ? previewHeight : previewWidth;
            int imageHeight = detector.isRotated(sensorOrientation) ? previewWidth : previewHeight;

            float scalePreview = Math.min(canvas.getWidth() / (float)imageWidth, canvas.getHeight() / (float)imageHeight);

            ArrayList<PointF> adjusted = new ArrayList<>();
            synchronized (lockPoints) {
                if (points != null) {
                    for (int i = 0; i < points.size(); i++) {
                        adjusted.add(new PointF(
                                (points.get(i).x - imageWidth / 2) * scalePreview + canvas.getWidth() / 2,
                                (points.get(i).y - imageHeight / 2) * scalePreview + canvas.getHeight() / 2));
                    }
                }
            }
            detector.drawScorecardBox(canvas, adjusted, 5, Color.RED);
        });

        findViewById(R.id.capture).setOnClickListener(view -> capture());
    }

    @Override
    protected void processImage() {
        ++timestamp;
        final long currTimestamp = timestamp;
        trackingOverlay.postInvalidate();

        // No mutex needed as this method is not reentrant.
        if (computingDetection) {
            readyForNextImage();
            return;
        }
        computingDetection = true;
        LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

        synchronized (lockFrameBitmap) {
            rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);
        }

        readyForNextImage();

        runInBackground(() -> {
            LOGGER.i("Running detection on image " + currTimestamp);
            final long startTime = SystemClock.uptimeMillis();

            ArrayList<PointF> _points = detector.inference(rgbFrameBitmap, sensorOrientation, false);

            if (_points != null) {
                synchronized (lockPoints) {
                    points = _points;
                }
            }

            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

            trackingOverlay.postInvalidate();

            computingDetection = false;

            runOnUiThread(() -> {
                showFrameInfo(previewWidth + "x" + previewHeight);
                showCropInfo(detector.getInputWidth() + "x" + detector.getInputHeight());
                showInference(lastProcessingTimeMs + "ms");
            });
        });
    }

    @Override
    protected int getLayoutId() {
        return R.layout.camera_connection_fragment_tracking;
    }

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }

    @Override
    protected void setNumThreads(int numThreads) {

    }

    @Override
    protected void setUseNNAPI(boolean isChecked) {

    }

    @Override
    public void onClick(View view) {
    }

    private ScoreCard sendScorecardImage() {
        if (rgbFrameBitmap == null) {
            runOnUiThread(() -> Toast.makeText(MainActivity.this, "Invalid camera preview", Toast.LENGTH_SHORT).show());
            return null;
        }

        Bitmap bitmapScorecard = null;
        ArrayList<PointF> _points = null;
        synchronized (lockFrameBitmap) {
            bitmapScorecard = rgbFrameBitmap.copy(rgbFrameBitmap.getConfig(), true);
        }

        if ( bitmapScorecard == null) {
            return null;
        }
        synchronized (lockPoints) {

            _points = new ArrayList<>(points);
        }

        try {
            return detector.sendScorecard(bitmapScorecard, _points,"kakaovx01", "ryon");
        }
        catch (Exception ex) {
            ex.printStackTrace();

            runOnUiThread(() -> Toast.makeText(MainActivity.this, ex.getMessage(), Toast.LENGTH_LONG).show());
        }

        return null;
    }

    private void capture() {
        new Thread(() -> {
            runOnUiThread(() -> {
                findViewById(R.id.progressbar).setVisibility(View.VISIBLE);
                findViewById(R.id.capture).setEnabled(false);
            });

            final ScoreCard scoreCard = sendScorecardImage();

            runOnUiThread(() -> {
                if (scoreCard != null)
                    displayScorecard(scoreCard);
                findViewById(R.id.progressbar).setVisibility(View.GONE);
                findViewById(R.id.capture).setEnabled(true);
            });
        }).start();
    }

    private void displayScorecard(ScoreCard scoreCard) {
        if (scoreCard == null)
            return;

        View view = View.inflate(this, R.layout.dialog_body, null);
        ((EditText) view.findViewById(R.id.edit_message)).setText(scoreCard.getSummary());

        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setView(view)
                .setPositiveButton("OK", new DialogInterface.OnClickListener() {
                    public void onClick(DialogInterface dialog, int id) {
                    }
                });

        builder.create().show();
    }
}