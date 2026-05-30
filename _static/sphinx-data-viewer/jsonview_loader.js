$(document).ready(function() {
  $("div.sdv-data").each(function(index){
    let data = $(this).attr('data-sdv')
    let expand = $(this).attr('data-expand')

    try {
      let tree = JsonView.createTree(data);
      JsonView.render(tree, $(this)[0]);
      if(expand==="True") {
        JsonView.expandChildren(tree);
      }
    } catch(e) {
      $(this).html('<p>Error: Could not parse json data</p>')
    }
  })




} );
