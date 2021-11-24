import numpy as np
import joblib as jbl
import os
import locale
from flask import Flask, request, render_template, make_response, url_for

# Runing application
app = Flask( __name__, static_url_path = '', template_folder = '' )

# Load model
model = jbl.load( 'model.pkl' )

"""
@api {get} /get_price
@apiName GetPrice

@apiParam {size}  only int.
@apiParam {rooms}  only int.
@apiParam {bathroom}  only int.
@apiParam {suite}  only int.
@apiParam {parking_spaces}  only int.


"""


@app.route( '/' )
def display_gui():
  return render_template( 'form.html' )

@app.route( '/get_price', methods=['POST'] )



def get_price():
      
  area = request.form['size']
  rooms = request.form['rooms']
  bathroom = request.form['bathroom']
  suite = request.form['suite']
  parking_spaces = request.form['parking_spaces']
# #   price_sell = request.form['price_sell']
# #   property_tax = request.form['property_tax']
  
  # print(area)
  params = np.array( [ [ area, rooms, bathroom,suite, parking_spaces] ] )
#   print( params )

  price = model.predict( params )[0]
#   print( "Estimated price: {}".format( str( price ) ) )
  
  return render_template( 'form.html', price = str( "O valor estimado para o imóvel é de R${:,.0f}".format(price)).replace(",",".")  )
  # return render_template( 'form.html', price = str( 1 ) )

if __name__ == "__main__":
  port = int( os.environ.get( 'PORT', 5500 ) )
  app.run( host='localhost', port=port )