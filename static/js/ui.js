var formdata = {"datasetname": "", "retrainmodel": "", "forecastvalue": "", "timeframe": ""}

var opens = []
var highs = []
var lows = []
var closes = []
var dates = []

var domain = 'localhost'
var port = 5000

var socket = io.connect('http://' + domain + ':' + port);

updatechart()

socket.on('connect', function() {
     console.log('connected');
});

socket.on('forecast', function(msg){
//    console.log(msg['data']['open'][0][0])
    opens = msg['data']['open'][0]
    highs = msg['data']['high'][0]
    lows = msg['data']['low'][0]
    closes = msg['data']['close'][0]
    dates = msg['data']['dates']
    console.log(msg)
    updatechart()
});

socket.on('progressbar', function(msg){
    $('#progressbarbar').css("width", msg['progress'])
    $('#progressbarbar').get(0).innerHTML = msg['progress']
    if (msg['progress'] == "100%"){
        $('#subloading').addClass('visually-hidden');
        $('#subvalidate').removeClass('visually-hidden');
        $('#container').removeClass('visually-hidden');
    }
});

socket.on('modelexistornot', function(msg){
    $('#loadtext').get(0).innerHTML = msg['exist']
});

function newforecast(val) {
    $('#forecast').get(0).value = val
}

$("#minutes").click(function(){
    $("#dropdownMenuButton1").get(0).innerText = "Minutes"
    $('#basic-addon1').get(0).innerHTML = "Minutes"
    formdata.timeframe = "minutes"
});

$("#hours").click(function(){
    $("#dropdownMenuButton1").get(0).innerText = "Hours"
    $('#basic-addon1').get(0).innerHTML = "Hours"
    formdata.timeframe = "hours"
});

$("#days").click(function(){
    $("#dropdownMenuButton1").get(0).innerText = "Days"
    $('#basic-addon1').get(0).innerHTML = "Days"
    formdata.timeframe = "days"
});

$("#form1").submit(function (e) {
     e.preventDefault();
     $('#subvalidate').addClass('visually-hidden');
     $('#subloading').removeClass('visually-hidden');
     $('#progressbarbar').css("width", "0%")
     $('#progressbar').removeClass('visually-hidden');
    /* Get checkbox is checked or not */
    if ($('#flexCheckChecked').get(0).checked == true){
//        $('#flexCheckChecked').get(0).value = true
        $('#loadtext').get(0).innerHTML = "Training new model..."
        formdata.retrainmodel = "true"
    }
    else{
//        $('#flexCheckChecked').get(0).value = false
        $('#loadtext').get(0).innerHTML = "Predicting..."
        formdata.retrainmodel = "false"
        };
    formdata.datasetname = $('#formFile').get(0).value.replace(/C:\\fakepath\\/i, '')
    formdata.forecastvalue = $('#forecast').get(0).value
    console.log(formdata)
    socket.emit("submit", formdata);

});
