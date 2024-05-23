"use client"

import { Button } from "@/components/ui/button";
import { Dotting, type DottingRef, useDotting, type PixelModifyItem, useData } from "dotting";
import { CircleHelp, Eraser } from "lucide-react";
import { useRef, useState } from "react";

type ApiData = {
  guess: number,
  confidence: number
}

export default function Home() {
  const zeros : number[] = Array(784).fill(0);
  const [data, setData] = useState<number[]>(zeros);
  const [guess, setGuess] = useState<number>(-1);
  const [confidence, setConfidence] = useState<number>(0.0);
  const ref = useRef<DottingRef>(null);
  const { clear } = useDotting(ref);

  const CreateEmptySquareData = (
    size: number,
  ): Array<Array<PixelModifyItem>> => {
    const data: Array<Array<PixelModifyItem>> = [];
    for (let i = 0; i < size; i++) {
      const row: Array<PixelModifyItem> = [];
      for (let j = 0; j < size; j++) {
        row.push({ rowIndex: i, columnIndex: j, color: "" });
      }
      data.push(row);
    }
    return data;
  };

  const {dataArray} = useData(ref);

  function displayData() {
    // transform data to a 784 array
    const dataFloat : number[] = [];
    for (let i = 0; i < 28; i++) {
      for (let j = 0; j < 28; j++) {
        dataFloat.push(dataArray[i][j].color === "#FFF" ? 1 : 0);
      }
    }

    // send data to the server
    // the date is : {image : [array of 784 floats]}
    fetch("http://localhost:8000/predict/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({image: dataFloat})
    }).then(response => response.json())
    .then((data: ApiData) => {
      setGuess(data.guess);
      setConfidence(data.confidence);
    })

    setData(dataFloat);
    console.log(dataFloat);
  }
  
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-10">
      <div className="flex flex-col items-center space-y-3">
      <h1 className="text-3xl font-bold">MNIST Classifier - Guess the number</h1>
      <p>Draw a number between 0 and 9</p>
        <Dotting width={500} ref={ref} height={500} brushColor="#FFF" defaultPixelColor="#000000" gridStrokeColor="#AAA" isGridFixed gridSquareLength={100} initLayers={[
        {
          id: "default",
          data: CreateEmptySquareData(28),
        }
      ]}/>
      <div className=" space-x-3">
        <Button type="button" onClick={clear} className=" space-x-1">
          <Eraser />
          <p> Clear</p>
        </Button>
        <Button type="button" onClick={displayData} className="space-x-1">
          <CircleHelp />
          <p>Guess</p>
        </Button>
      </div>

        
      </div>
      <div className="flex flex-col items-center">
        <h2 className="text-2xl font-bold">Result</h2>
        <p>Guess: {guess}</p>
        <p>Confidence: {confidence}</p>
      </div>

    </main>
  );
}
