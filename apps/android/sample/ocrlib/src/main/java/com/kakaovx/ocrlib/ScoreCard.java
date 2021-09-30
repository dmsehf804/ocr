package com.kakaovx.ocrlib;

import java.util.ArrayList;

public class ScoreCard {
    public String userId;
    public String userName;

    public int resultCode;
    public String resultMessage;

    public String clubName;
    public ArrayList<String> courseNames;
    public String date;
    public String teeoff;

    public ArrayList<Integer> pars;

    public class ScoreInfo {
        public String playerName;
        public ArrayList<Integer> scores;
        public int totalScore;

        public ScoreInfo() {
            scores = new ArrayList<>();
        }
    }

    public ArrayList<ScoreInfo> scoreInfos;

    public ScoreCard() {
        courseNames = new ArrayList<>();
        pars = new ArrayList<>();
        scoreInfos = new ArrayList<>();
    }

    public String getSummary() {
        StringBuilder sb = new StringBuilder();

        if (resultCode == 1) {
            sb.append("result : " + clubName + "\n");

        }
        else {
            sb.append(resultMessage);
        }

        return sb.toString();
    }
}
