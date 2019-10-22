var socket = io();
$('#chat').on('submit', function(e){
  socket.emit('send message', $('#name').val(), $('#message').val());
  $('#message').val('');
  $("#message").focus();
  e.preventDefault();
});

socket.on('receive message', function(msg){
  $('#chatLog').append(msg+'\n');
  var JSAlert = require('js-alert');
  JSAlert.alert(msg);
  $('#chatLog').scrollTop($('#chatLog')[0].scrollHeight);
});

socket.on('change name', function(name){
  $('#name').val(name);
});

var showPopUp= function()=>{
  console.log("gd");
  // __dirname + '/FrontEnd/chattingPage.html'
  var popUpUrl ="/draw_page/draw.html";
  var popUpOption = "width=1000, height=500 resiresizable=no, scrollbars=no, status=no;";

  window.open(popUpUrl,"",popUpOption);
}
