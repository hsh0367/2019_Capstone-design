var express = require('express');
var app = express();
var http = require('http').Server(app);
var io = require('socket.io')(http);

//모든 request는 cleint를 responce하도록
app.get('/',function(req, res){
  res.sendFile(__dirname + '/FrontEnd/cheatingPage.html');
});

//io.on(EVENT,함수)는 서버에 전달된 EVENT를 인식하여 함수를 실행시키는 event listener
var count = 1;
io.on('connection', function(socket){
  console.log('user connected: ', socket.id);
  var name = "user" + count++;
  io.to(socket.id).emit('change name', name);

  socket.on('disconnect', function(){
      console.log('user disconnect : '+ socket.id);
  });

  socket.on('send message', function(name,text){
    var msg = name + ':' + text;
    io.emit('receive message', msg);
    console.log(msg);
    // alert(msg);
  });

});

http.listen(3003, function(){
  console.log('server on!');
});

// server.listen(80, ()=>{
//     console.log('HTTP server listen on port 80');
// });
//
// io.on('connection',(socketServer)=>{
//   socketServer.on('npmStop',()=>{
//     process.exit(0);
//   });
// });
