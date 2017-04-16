/*
Contains the code for rendering article results.
 */

// This makes Javascript run in strict mode, which adds some restrictions that make it generally more sane.
"use strict";

$(document).ready(function(){
    showFinalScore();
    showVerdict();
    computeProgressionClasses();
});

//displays the final score
function showFinalScore(){
    document.getElementById("finalScore").innerHTML += score;
}

//determines which verdict to show
function showVerdict(){
    document.getElementById("trueVerdict").hidden = !verdict;
    document.getElementById("falseVerdict").hidden = verdict;
}

//processes each content word and gives it a style class depending on its word_score
function computeProgressionClasses(){
    var contentArray = content.split(/\s+/);
    for(var i = 0; i < word_scores.length; i++){
        var wordClass;

        //assigns a class based on the score for the word
        if(word_scores[i] <= .1){
            wordClass = "zero";
        }else if(word_scores[i] <= .2){
            wordClass = "one";
        }else if(word_scores[i] <= .3){
            wordClass = "two";
        }else if(word_scores[i] <= .4){
            wordClass = "three";
        }else if(word_scores[i] <= .5){
            wordClass = "four";
        }else if(word_scores[i] <= .6){
            wordClass = "five";
        }else if(word_scores[i] <= .7){
            wordClass = "six";
        }else if(word_scores[i] <= .8){
            wordClass = "seven";
        }else if(word_scores[i] <= .9){
            wordClass = "eight";
        }else{
            wordClass = "nine";
        }

        contentArray[i] += " ";
        document.getElementById("progressionContainer").innerHTML += "<span class=" + wordClass + ">"+contentArray[i]+"</span>";
    }
}