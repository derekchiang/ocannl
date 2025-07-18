open Base
open Ocannl
open Stdio
module Tn = Ir.Tnode
module IDX = Train.IDX
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL
module CDSL = Train.CDSL
module Rand = Ir.Rand.Lib

module type Backend = Ir.Backend_intf.Backend

let read_names () = In_channel.read_lines "bin/names.txt"

let bigrams s =
  let chars = String.to_list s in
  let front = '.' :: chars in
  let back = chars @ [ '.' ] in
  List.zip_exn front back

let get_all_bigrams () = List.(read_names () >>| bigrams |> concat)
let letters = List.init 26 ~f:(fun i -> Char.of_int_exn (Char.to_int 'a' + i))
let letters_with_dot = '.' :: letters

let char_to_index_tbl =
  let tbl = Hashtbl.create (module Char) in
  List.iteri letters_with_dot ~f:(fun i c -> Hashtbl.set tbl ~key:c ~data:i);
  tbl

let char_index c =
  match Hashtbl.find char_to_index_tbl c with
  | Some i -> i
  | None -> failwith (Printf.sprintf "Character not found: %c" c)

let bigrams_to_indices bigrams = List.(bigrams >>| fun (c1, c2) -> (char_index c1, char_index c2))
let print_tensor t = Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default t

let one_hot ~num_classes indices =
  let num_classes = num_classes - 1 in
  let%op classes = TDSL.range num_classes in
  (* Broadcast indices to add a dimension for classes *)
  let%op indices_expanded = indices ++ "i => i j" in
  (* Broadcast classes to match indices dimension *)
  let%op classes_expanded = classes ++ "j => i j" in
  (* Compare expanded tensors *)
  let%op one_hot = indices_expanded = classes_expanded in
  one_hot

let _print_range_tensor () =
  let module Backend = (val Backends.fresh_backend ()) in
  let stream = Backend.(new_stream @@ get_device ~ordinal:0) in
  let ctx = Backend.make_context stream in
  let upto = 5 in
  let num_classes = 3 in
  let%op tensor1 = TDSL.range upto in
  let%op classes = TDSL.range num_classes in

  (* Test different broadcasting approaches *)
  let%op indices_expanded = tensor1 ++ "i => i j" in
  let%op classes_expanded = classes ++ "j => i j" in
  let%op comparison = indices_expanded = classes_expanded in

  printf "Testing one-hot encoding with broadcasting:\n";
  printf "Indices: 0,1,2,3,4  Classes: 0,1,2\n";
  printf "Expected: each row should be one-hot for that index\n";
  Train.forward_and_forget (module Backend) ctx comparison;
  print_tensor comparison

(* let () = print_range_tensor () *)

let () =
  Rand.init 0;

  let module Backend = (val Backends.fresh_backend ()) in
  let stream = Backend.(new_stream @@ get_device ~ordinal:0) in
  let ctx = Backend.make_context stream in
  let bigrams = get_all_bigrams () |> bigrams_to_indices in

  let test_bigrams = List.take bigrams 10 in
  let int_input, _ = List.unzip test_bigrams in
  let input_list = int_input |> List.map ~f:Float.of_int in
  let input_arr = Array.of_list input_list in
  let size = Array.length input_arr in

  printf "First 10 character indices: %s\n"
    (Array.to_list input_arr |> List.map ~f:Float.to_string |> String.concat ~sep:", ");

  let tensor = TDSL.ndarray ~output_dims:[ size ] input_arr in
  let one_hot_tensor = one_hot ~num_classes:27 tensor in

  (* Mark the tensor to be on host for printing *)
  (* Train.set_on_host one_hot_tensor.value; *)

  (* Compute the tensor *)
  let routine = Train.to_routine (module Backend) ctx IDX.empty @@ Train.forward one_hot_tensor in
  Train.run routine;
  printf "One-hot encoding (first 10 chars, all 27 classes):\n";
  print_tensor one_hot_tensor
