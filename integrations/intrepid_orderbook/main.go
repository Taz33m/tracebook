package main

import "os"

func main() {
	os.Exit(serve(os.Stdin, os.Stdout))
}
