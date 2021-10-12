package com.kakaovx.ocrlib;

import android.graphics.Bitmap;
import android.graphics.PointF;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.concurrent.TimeUnit;

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class CommHelper {
    private final String url = "http://10.12.200.137:5000/ocr";

    public ScoreCard sendScorecard(Bitmap bitmap,  String userId, String userName) throws IOException {
        if (bitmap == null)
            return null;

        ByteArrayOutputStream stream = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream);
        byte[] byteArray = stream.toByteArray();

        final OkHttpClient client = new OkHttpClient.Builder()
                .readTimeout(30, TimeUnit.SECONDS)
                .writeTimeout(30, TimeUnit.SECONDS)
                .build();
        System.out.println("bbbbbbbbbbbbbbb@@@@@");
        RequestBody requestBody = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("user_id", userId)
                .addFormDataPart("user_name", userName)
                .addFormDataPart("image", "image.png", RequestBody.create(MediaType.parse("image/*png"), byteArray))
                .build();
        Request request = new Request.Builder()
                .url(url)
                .addHeader("Content-Type", " application/x-www-form-urlencoded")
                .post(requestBody)
                .build();
        System.out.println("cccccccccc~~~~~~~~~~");
        try (Response response = client.newCall(request).execute()) {
            System.out.println("ceeeeeeeeeeeeeeeddd~~~~~~~~");
            if (response.isSuccessful() == false)
                return null;
            System.out.println("cddddddddddddddd~~~~~~~~");
            return parseResponse(response.body().string());
        }
    }

    private ScoreCard parseResponse(String json) {
        ScoreCard scoreCard = new ScoreCard();
        scoreCard.resultCode = 1;

        try {
            JSONObject jsonObject = new JSONObject(json);
            if (jsonObject == null) {
                scoreCard.resultCode = 0;
                scoreCard.resultMessage = "Not in json format";
                return scoreCard;
            }

            int resultCode = jsonObject.getInt("result_code");
            if (resultCode != 0) {
                scoreCard.resultCode = resultCode;
                scoreCard.resultMessage = "An error was returned";
                return scoreCard;
            }

            JSONObject scoreInfo = jsonObject.getJSONObject("score_info");
            if (scoreInfo == null) {
                scoreCard.resultCode = 0;
                scoreCard.resultMessage = "Invalid score_info";
                return scoreCard;
            }

            scoreCard.clubName = scoreInfo.getString("club_name");


        }
        catch (Exception ex) {
            ex.printStackTrace();

            scoreCard.resultCode = 0;
            scoreCard.resultMessage = ex.getMessage();
        }

        return scoreCard;
    }
}
