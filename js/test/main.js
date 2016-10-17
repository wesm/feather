d3.request("feather/iris.feather")
    .responseType("arraybuffer")
    .get(function(error, data) {
        if (error === null) {
            var df = convertToDataFrame(data.response);

            df.SepalLength = df["Sepal.Length"];
            df.PetalLength = df["Petal.Length"];

            // d3 is mildly awkward with struct of arrays - oh well
            var dummy = new Uint8Array(new ArrayBuffer(df.SepalLength.length));

            var xExtent = d3.extent(df.SepalLength);
            var yExtent = d3.extent(df.PetalLength);
            var xScale = d3.scaleLinear().domain(xExtent).range([10, 490]);
            var yScale = d3.scaleLinear().domain(yExtent).range([490, 10]);
            var colorScale = d3.scaleOrdinal(d3.schemeCategory10);
            d3.select("#main")
                .append("svg")
                .attr("width", 500)
                .attr("height", 500)
                .selectAll("circle")
                .data(dummy)
                .enter()
                .append("circle")
                .attr("cx", function(d,i) { return xScale(df.SepalLength[i]); })
                .attr("cy", function(d,i) { return yScale(df.PetalLength[i]); })
                .attr("fill", function(d, i) { return colorScale(df.Species[i]); })
                .attr("r", 5);
        }
    });
