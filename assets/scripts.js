$(document).ready(function () {
    $('#lcs_result').hide();
    $('#dl_result').hide();
    $("#bt").click(function () {
        // $.get('http://localhost:8007/dl_results', function (data, status) {
        //     console.log(data.result)
        $.post("http://localhost:8007/lcs_results",
            {
                "title": $('#title').val()
                    // $('#title').val()
            },
            function (data, status) {
                console.log( data.result);
                var i=0
                re =  data.result;
                var res='<h3>lcs results:</h3>'
                for (i=0; i<re.length; i++){
                    res+=re[i]+'<br>'
                }
                $('#lcs_result').html()=res
                $('#lcs_result').show()
            });
            $.post("http://localhost:8007/dl_results",
            {
                "title": $('#title').val()
                    // $('#title').val()
            },
            function (data, status) {
                console.log( data.result);
                re =  data.result;
                var i=0
                var res='<h3>dl results:</h3>'
                for (i=0; i<re.length; i++){
                    res+=re[i]+'<br>'
                }
                $('#dl_result').html()=res
                $('#dl_result').show()
            });
    });
});
;


