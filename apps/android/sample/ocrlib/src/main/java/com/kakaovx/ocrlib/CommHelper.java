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

    private final String dummyResponse =
            "{'user_id': 'kakaovx01', 'user_name': 'ryon', 'result_code': 1, 'score_info': {'club_name': '', 'course_name': ['HMl', 'Forest'], 'user_name': '', 'date': '2020/09/05', 'teeoff': 'PM1:55', 'par': ['4', '4', '5', '4', '3', '5', '5', '3', '4', '4', '4', '3', '4', '3', '4', '5', '3', '5'], 'player_infos': [{'player_name': 'Ox', 'scores': ['2', '1', '5', '0', '1', '4', '1', '3', '1', '2', '3', '3', '3', '-1', '2', '3', '0', '0'], 'total_score': '105'}, {'player_name': '이x', 'scores': ['2', '4', '3', '1', '2', '2', '4', '2', '1', '2', '2', '2', '1', '1', '3', '4', '0', '0'], 'total_score': '108'}, {'player_name': '조혜연', 'scores': ['3', '1', '2', '0', '1', '4', '3', '2', '1', '3', '3', '0', '1', '2', '1', '1', '0', '0'], 'total_score': '100'}, {'player_name': '우', 'scores': ['3', '1', '5', '0', '2', '3', '3', '2', '1', '2', '3', '1', '3', '2', '3', '2', '1', '0'], 'total_score': '109'}]}}";

    public ScoreCard sendScorecardDummy(Bitmap bitmap, String userId, String userName) throws IOException {
        return parseResponse(dummyResponse);
    }

    public ScoreCard sendScorecard(Bitmap bitmap, ArrayList<PointF> points, String userId, String userName) throws IOException {
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
